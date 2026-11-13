// sherpa-onnx/csrc/ctc-prefix-search.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/ctc-prefix-search.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

constexpr float kLogZero = -std::numeric_limits<float>::infinity();

float LogAdd(float a, float b) {
  if (a == kLogZero) return b;
  if (b == kLogZero) return a;
  float high = std::max(a, b);
  float low = std::min(a, b);
  return high + std::log1p(std::exp(low - high));
}

size_t RoundUpTo8(size_t n) { return (n + 7) & ~static_cast<size_t>(7); }

}  // namespace

// The CTC recurrence's state for one prefix.
struct CtcPrefixSearch::CtcState {
  // Alignment mass ending in blank and in the last token, through the last
  // finished frame. Meaningful while the prefix is in the beam.
  float log_blank;
  float log_nonblank;
  // The same for the frame in progress, valid when stamp == frame + 1.
  float next_blank;
  float next_nonblank;
  int32_t stamp;
  int32_t frame;     // Frame whose emission of the last token weighed most.
  float emit_score;  // That emission's log mass.
};

// Each of a prefix's two endings (in blank, in its last token) has its own
// alignments and so its own need; see NoteContribution.
struct CtcPrefixSearch::SurvivalTrace {
  SurvivalNeed blank;
  SurvivalNeed nonblank;
  SurvivalNeed next_blank;
  SurvivalNeed next_nonblank;
  float best_blank_term;
  float best_nonblank_term;
};

CtcPrefixSearch::CtcPrefixSearch(
    const CtcPrefixSearchOptions &options,
    std::shared_ptr<const CtcSearchVocabulary> vocabulary,
    std::vector<Scorer *> scorers, std::vector<float> scorer_weights)
    : options_(options),
      survival_offset_(RoundUpTo8(sizeof(CtcState))),
      trie_(std::move(vocabulary), std::move(scorers),
            std::move(scorer_weights),
            RoundUpTo8(sizeof(CtcState)) +
                (options.trace_survival ? sizeof(SurvivalTrace) : 0),
            options.word_boundary_char) {
  recombine_ = (options_.alignment == CtcAlignmentMode::kMax
                    ? options_.recombine
                    : options_.prune_dominated_in_sum) &&
               trie_.ScorersRecombine();

  Node *root = trie_.At(ScoredPrefixTrie::kRoot);
  *Ctc(root) = {0.0f, kLogZero, kLogZero, kLogZero, 0, 0, kLogZero};
  root->pinned = true;
  if (options_.trace_survival) {
    *SurvivalOf(root) = {{0.0f, 1}, {0.0f, 1}, {0.0f, 1},
                         {0.0f, 1}, kLogZero,  kLogZero};
  }
  active_.push_back(ScoredPrefixTrie::kRoot);
}

CtcPrefixSearch::~CtcPrefixSearch() = default;

CtcPrefixSearch::CtcState *CtcPrefixSearch::Ctc(Node *node) const {
  return reinterpret_cast<CtcState *>(trie_.SearchState(node));
}

CtcPrefixSearch::SurvivalTrace *CtcPrefixSearch::SurvivalOf(Node *node) const {
  return reinterpret_cast<SurvivalTrace *>(trie_.SearchState(node) +
                                           survival_offset_);
}

// Folds the need of one term's alignment into an ending's need for the
// frame in progress. In max mode the ending holds its largest term, so it
// takes that term's need. In sum mode every term is part of the ending, so
// it takes the most demanding.
void CtcPrefixSearch::NoteContribution(float term, const SurvivalNeed &from,
                                       float *best_term,
                                       SurvivalNeed *next) const {
  if (options_.alignment == CtcAlignmentMode::kMax) {
    if (term > *best_term) *next = from;
  } else if (*best_term == kLogZero) {
    *next = from;
  } else {
    next->gap = std::max(next->gap, from.gap);
    next->rank = std::max(next->rank, from.rank);
  }
  *best_term = std::max(*best_term, term);
}

float CtcPrefixSearch::Accumulate(float sum, float term) const {
  return options_.alignment == CtcAlignmentMode::kSum ? LogAdd(sum, term)
                                                      : std::max(sum, term);
}

float CtcPrefixSearch::Acoustic(float blank, float nonblank) const {
  return Accumulate(blank, nonblank);
}

float CtcPrefixSearch::LengthBonus(const Node &node, bool at_end) const {
  if (options_.length_bonus == 0) return 0;
  return options_.length_bonus *
         trie_.LengthIn(node, options_.length_bonus_unit, at_end);
}

void CtcPrefixSearch::Touch(uint32_t index) {
  Node *node = trie_.At(index);
  CtcState *state = Ctc(node);
  if (state->stamp == frame_ + 1) return;
  state->stamp = frame_ + 1;
  state->next_blank = kLogZero;
  state->next_nonblank = kLogZero;
  if (options_.trace_survival) {
    SurvivalOf(node)->best_blank_term = kLogZero;
    SurvivalOf(node)->best_nonblank_term = kLogZero;
  }
  touched_.push_back(index);
}

void CtcPrefixSearch::BeginFrame(const float *log_probs) {
  touched_.clear();
  trie_.StartScoring();

  // The most probable emittable tokens, ties to the lower ID.
  const SearchVocabulary &vocabulary = trie_.Vocabulary();
  int32_t blank = vocabulary.blank_id;
  int32_t vocab_size = static_cast<int32_t>(vocabulary.tokens.size());
  size_t limit =
      static_cast<size_t>(std::max(options_.max_token_candidates, 0));
  candidates_.clear();
  for (int32_t token = 0; token != vocab_size; ++token) {
    if (token == blank || !vocabulary.tokens[token].emit) continue;
    if (candidates_.size() == limit &&
        (limit == 0 || !(log_probs[token] > log_probs[candidates_.back()]))) {
      continue;
    }
    auto position = std::upper_bound(candidates_.begin(), candidates_.end(),
                                     token, [log_probs](int32_t a, int32_t b) {
                                       return log_probs[a] > log_probs[b];
                                     });
    candidates_.insert(position, token);
    if (candidates_.size() > limit) candidates_.pop_back();
  }

  for (uint32_t index : active_) {
    Node *node = trie_.At(index);
    float log_blank = Ctc(node)->log_blank;
    float log_nonblank = Ctc(node)->log_nonblank;
    float total = Acoustic(log_blank, log_nonblank);
    int32_t last = node->token;

    // The need of the alignment each of this prefix's sources stands for.
    SurvivalNeed blank_need = {0, 0}, nonblank_need = {0, 0},
                 total_need = {0, 0};
    if (options_.trace_survival) {
      blank_need = SurvivalOf(node)->blank;
      nonblank_need = SurvivalOf(node)->nonblank;
      if (options_.alignment == CtcAlignmentMode::kMax) {
        total_need = log_blank >= log_nonblank ? blank_need : nonblank_need;
      } else if (log_blank == kLogZero) {
        total_need = nonblank_need;
      } else if (log_nonblank == kLogZero) {
        total_need = blank_need;
      } else {
        total_need = {std::max(blank_need.gap, nonblank_need.gap),
                      std::max(blank_need.rank, nonblank_need.rank)};
      }
    }

    Touch(index);
    CtcState *state = Ctc(node);
    float stay_blank = total + log_probs[blank];
    state->next_blank = Accumulate(state->next_blank, stay_blank);
    if (options_.trace_survival) {
      SurvivalTrace *trace = SurvivalOf(node);
      NoteContribution(stay_blank, total_need, &trace->best_blank_term,
                       &trace->next_blank);
    }
    if (last >= 0) {
      float stay_nonblank = log_nonblank + log_probs[last];
      state->next_nonblank = Accumulate(state->next_nonblank, stay_nonblank);
      if (options_.trace_survival) {
        SurvivalTrace *trace = SurvivalOf(node);
        NoteContribution(stay_nonblank, nonblank_need,
                         &trace->best_nonblank_term, &trace->next_nonblank);
      }
    }
    for (int32_t token : candidates_) {
      // Repeating the last token extends the prefix only after a blank.
      float source = token == last ? log_blank : total;
      if (source == kLogZero) continue;
      bool created = false;
      uint32_t child_index = trie_.FindOrCreateChild(index, token, &created);
      Node *child = trie_.At(child_index);
      if (created) {
        *Ctc(child) = {kLogZero, kLogZero, kLogZero, kLogZero,
                       0,        frame_,   kLogZero};
      }
      Touch(child_index);
      CtcState *child_state = Ctc(child);
      float emitted = source + log_probs[token];
      child_state->next_nonblank =
          Accumulate(child_state->next_nonblank, emitted);
      if (emitted > child_state->emit_score) {
        child_state->emit_score = emitted;
        child_state->frame = frame_;
      }
      if (options_.trace_survival) {
        SurvivalTrace *trace = SurvivalOf(child);
        NoteContribution(emitted, token == last ? blank_need : total_need,
                         &trace->best_nonblank_term, &trace->next_nonblank);
      }
    }
  }
}

bool CtcPrefixSearch::GatherScorerWave(
    int32_t scorer, std::vector<ScorerExtension> *extensions) {
  return trie_.GatherScorerWave(scorer, extensions);
}

void CtcPrefixSearch::ApplyScorerWave(int32_t scorer,
                                      const ScorerAdvanceResult *results) {
  trie_.ApplyScorerWave(scorer, results);
}

// Puts the hypotheses to keep at the front of "order" (positions in
// touched_), best first, and returns how many: the best "limit" within the
// score beam, passing over each one that a kept hypothesis with the same
// future (last token, word state, and scorer states) matches or beats both
// ending in blank and ending in its last token. Everything scored from
// here on is the same for both, so the one passed over cannot win. Only
// hypotheses that reach the front of the ranking are examined.
size_t CtcPrefixSearch::KeepUndominated(std::vector<uint32_t> *order,
                                        size_t limit) {
  auto worse = [this](uint32_t a, uint32_t b) {
    if (touched_scores_[a] != touched_scores_[b]) {
      return touched_scores_[a] < touched_scores_[b];
    }
    return touched_[a] > touched_[b];
  };
  std::make_heap(order->begin(), order->end(), worse);
  auto heap_end = order->end();

  // Kept hypotheses chained by a hash of their future.
  size_t table_size = 16;
  while (table_size < 2 * limit) table_size *= 2;
  future_heads_.assign(table_size, ScoredPrefixTrie::kNone);
  future_chain_.clear();
  kept_.clear();
  dropped_.clear();

  float cutoff = kLogZero;
  while (kept_.size() != limit && heap_end != order->begin()) {
    std::pop_heap(order->begin(), heap_end, worse);
    uint32_t candidate = *--heap_end;
    float score = touched_scores_[candidate];
    if (score == kLogZero || score < cutoff) {
      ++heap_end;  // Neither it nor anything after it is kept.
      break;
    }
    if (kept_.empty()) cutoff = score - options_.beam;

    Node *b = trie_.At(touched_[candidate]);
    float b_rest = b->scorer_total + LengthBonus(*b, false);
    float b_blank = Ctc(b)->next_blank + b_rest;
    float b_nonblank = Ctc(b)->next_nonblank + b_rest;
    size_t slot =
        (trie_.HashScorerFuture(*b) ^ static_cast<uint32_t>(b->token)) &
        (table_size - 1);
    bool dominated = false;
    for (uint32_t k = future_heads_[slot]; k != ScoredPrefixTrie::kNone;
         k = future_chain_[k]) {
      Node *a = trie_.At(touched_[kept_[k]]);
      if (a->token != b->token || !trie_.SameScorerFuture(*a, *b)) continue;
      float a_rest = a->scorer_total + LengthBonus(*a, false);
      float a_blank = Ctc(a)->next_blank + a_rest;
      float a_nonblank = Ctc(a)->next_nonblank + a_rest;
      if (a_blank >= b_blank && a_nonblank >= b_nonblank) {
        dominated = true;
        break;
      }
      if (b_blank >= a_blank && b_nonblank >= a_nonblank) {
        // Possible only when their scores tie: b takes a's place.
        std::swap(kept_[k], candidate);
        dominated = true;
        break;
      }
    }
    if (dominated) {
      dropped_.push_back(candidate);
      ++num_recombined_;
      continue;
    }
    future_chain_.push_back(future_heads_[slot]);
    future_heads_[slot] = static_cast<uint32_t>(kept_.size());
    kept_.push_back(candidate);
  }

  // Kept, then passed over, then never reached.
  size_t keep = kept_.size();
  dropped_.insert(dropped_.end(), order->begin(), heap_end);
  std::copy(kept_.begin(), kept_.end(), order->begin());
  std::copy(dropped_.begin(), dropped_.end(), order->begin() + keep);
  return keep;
}

void CtcPrefixSearch::EndFrame() {
  trie_.FinishScoring();

  touched_scores_.resize(touched_.size());
  for (size_t i = 0; i != touched_.size(); ++i) {
    Node *node = trie_.At(touched_[i]);
    touched_scores_[i] =
        node->rejected
            ? kLogZero
            : Acoustic(Ctc(node)->next_blank, Ctc(node)->next_nonblank) +
                  node->scorer_total + LengthBonus(*node, false);
  }

  // Best first; equal scores keep the order of their records' indices.
  std::vector<uint32_t> order(touched_.size());
  std::iota(order.begin(), order.end(), 0);
  size_t keep = std::min(order.size(),
                         static_cast<size_t>(std::max(options_.max_active, 0)));
  if (recombine_) {
    keep = KeepUndominated(&order, keep);
  } else {
    auto better = [this](uint32_t a, uint32_t b) {
      if (touched_scores_[a] != touched_scores_[b]) {
        return touched_scores_[a] > touched_scores_[b];
      }
      return touched_[a] < touched_[b];
    };
    std::partial_sort(order.begin(), order.begin() + keep, order.end(), better);
    if (keep != 0) {
      float cutoff = touched_scores_[order[0]] - options_.beam;
      while (keep != 0 && (touched_scores_[order[keep - 1]] == kLogZero ||
                           touched_scores_[order[keep - 1]] < cutoff)) {
        --keep;
      }
    }
  }

  for (uint32_t index : active_) trie_.At(index)->pinned = false;
  active_.clear();
  for (size_t i = 0; i != keep; ++i) {
    uint32_t index = touched_[order[i]];
    Node *node = trie_.At(index);
    CtcState *state = Ctc(node);
    state->log_blank = state->next_blank;
    state->log_nonblank = state->next_nonblank;
    node->pinned = true;
    active_.push_back(index);
    if (options_.trace_survival) {
      // Being kept this frame is part of what both endings needed.
      SurvivalTrace *trace = SurvivalOf(node);
      float gap = touched_scores_[order[0]] - touched_scores_[order[i]];
      int32_t rank = static_cast<int32_t>(i) + 1;
      trace->blank = {std::max(trace->next_blank.gap, gap),
                      std::max(trace->next_blank.rank, rank)};
      trace->nonblank = {std::max(trace->next_nonblank.gap, gap),
                         std::max(trace->next_nonblank.rank, rank)};
    }
  }
  for (size_t i = keep; i != order.size(); ++i) {
    trie_.ReleaseIfUnused(touched_[order[i]]);
  }
  num_kept_ += static_cast<int64_t>(keep);
  ++frame_;
}

void CtcPrefixSearch::Decode(const float *log_probs, int32_t num_frames) {
  size_t vocab_size = trie_.Vocabulary().tokens.size();
  const std::vector<Scorer *> &scorers = trie_.Scorers();
  std::vector<ScorerExtension> extensions;
  std::vector<ScorerAdvanceResult> results;
  for (int32_t t = 0; t != num_frames; ++t) {
    BeginFrame(log_probs + t * vocab_size);
    for (size_t s = 0; s != scorers.size(); ++s) {
      extensions.clear();
      while (GatherScorerWave(s, &extensions)) {
        results.assign(extensions.size(), ScorerAdvanceResult());
        scorers[s]->Advance({extensions.data(), extensions.size()},
                            results.data());
        ApplyScorerWave(s, results.data());
        extensions.clear();
      }
    }
    EndFrame();
  }
}

CtcPrefixHypothesis CtcPrefixSearch::MakeHypothesis(
    uint32_t index, float score, float acoustic,
    const float *final_scores) const {
  CtcPrefixHypothesis hypothesis;
  hypothesis.score = score;
  hypothesis.acoustic_log_prob = acoustic;
  Node *node = trie_.At(index);
  const CtcState &state = *Ctc(node);
  if (options_.trace_survival) {
    const SurvivalTrace &trace = *SurvivalOf(node);
    bool blank_only = state.log_nonblank == kLogZero;
    bool nonblank_only = state.log_blank == kLogZero;
    SurvivalNeed need;
    if (options_.alignment == CtcAlignmentMode::kMax) {
      need =
          state.log_blank >= state.log_nonblank ? trace.blank : trace.nonblank;
    } else if (blank_only || nonblank_only) {
      need = blank_only ? trace.blank : trace.nonblank;
    } else {
      need = {std::max(trace.blank.gap, trace.nonblank.gap),
              std::max(trace.blank.rank, trace.nonblank.rank)};
    }
    hypothesis.required_score_beam = need.gap;
    hypothesis.required_max_active = need.rank;
  }
  for (size_t s = 0; s != trie_.NumScorers(); ++s) {
    hypothesis.scorer_log_scores.push_back(
        trie_.Subtotals(node)[s] + (final_scores ? final_scores[s] : 0));
  }
  trie_.TokensOf(index, &hypothesis.tokens);
  for (; node->parent != ScoredPrefixTrie::kNone;
       node = trie_.At(node->parent)) {
    hypothesis.timestamps.push_back(Ctc(node)->frame);
  }
  std::reverse(hypothesis.timestamps.begin(), hypothesis.timestamps.end());
  return hypothesis;
}

CtcPrefixHypothesis CtcPrefixSearch::Best() const {
  uint32_t best = ScoredPrefixTrie::kNone;
  float best_score = kLogZero;
  float best_acoustic = kLogZero;
  for (uint32_t index : active_) {
    Node *node = trie_.At(index);
    float acoustic = Acoustic(Ctc(node)->log_blank, Ctc(node)->log_nonblank);
    float score = acoustic + node->scorer_total + LengthBonus(*node, false);
    if (best == ScoredPrefixTrie::kNone || score > best_score) {
      best = index;
      best_score = score;
      best_acoustic = acoustic;
    }
  }
  if (best == ScoredPrefixTrie::kNone) return CtcPrefixHypothesis();
  return MakeHypothesis(best, best_score, best_acoustic, nullptr);
}

std::vector<CtcPrefixHypothesis> CtcPrefixSearch::Finish(int32_t nbest) {
  size_t num_scorers = trie_.NumScorers();
  std::vector<float> final_scores;
  std::vector<bool> accepted;
  trie_.FinalScores(active_, &final_scores, &accepted);

  std::vector<std::pair<float, size_t>> ranked;
  for (size_t i = 0; i != active_.size(); ++i) {
    if (!accepted[i]) continue;
    Node *node = trie_.At(active_[i]);
    float score = Acoustic(Ctc(node)->log_blank, Ctc(node)->log_nonblank) +
                  node->scorer_total + LengthBonus(*node, true);
    for (size_t s = 0; s != num_scorers; ++s) {
      score += trie_.Weights()[s] * final_scores[i * num_scorers + s];
    }
    ranked.emplace_back(score, i);
  }
  std::stable_sort(
      ranked.begin(), ranked.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });
  if (ranked.size() > static_cast<size_t>(std::max(nbest, 0))) {
    ranked.resize(std::max(nbest, 0));
  }

  std::vector<CtcPrefixHypothesis> hypotheses;
  for (const auto &entry : ranked) {
    Node *node = trie_.At(active_[entry.second]);
    hypotheses.push_back(
        MakeHypothesis(active_[entry.second], entry.first,
                       Acoustic(Ctc(node)->log_blank, Ctc(node)->log_nonblank),
                       final_scores.data() + entry.second * num_scorers));
  }
  return hypotheses;
}

void RunScorerWavesTogether(const std::vector<CtcPrefixSearch *> &searches,
                            const std::vector<Scorer *> &scorers) {
  std::vector<size_t> counts;
  std::vector<ScorerExtension> extensions;
  std::vector<ScorerAdvanceResult> results;
  for (size_t s = 0; s != scorers.size(); ++s) {
    while (true) {
      extensions.clear();
      counts.assign(searches.size(), 0);
      for (size_t k = 0; k != searches.size(); ++k) {
        size_t before = extensions.size();
        searches[k]->GatherScorerWave(s, &extensions);
        counts[k] = extensions.size() - before;
      }
      if (extensions.empty()) break;
      results.assign(extensions.size(), ScorerAdvanceResult());
      scorers[s]->Advance({extensions.data(), extensions.size()},
                          results.data());
      size_t offset = 0;
      for (size_t k = 0; k != searches.size(); ++k) {
        if (counts[k] != 0) {
          searches[k]->ApplyScorerWave(s, results.data() + offset);
        }
        offset += counts[k];
      }
    }
  }
}

void DecodeFramesTogether(const std::vector<CtcPrefixSearch *> &searches,
                          const std::vector<Scorer *> &scorers,
                          const float *log_probs, int32_t num_frames,
                          int32_t vocab_size, const int64_t *lengths) {
  std::vector<CtcPrefixSearch *> live;
  for (int32_t t = 0; t != num_frames; ++t) {
    live.clear();
    for (size_t i = 0; i != searches.size(); ++i) {
      if (lengths != nullptr && t >= lengths[i]) continue;
      searches[i]->BeginFrame(log_probs + (i * num_frames + t) * vocab_size);
      live.push_back(searches[i]);
    }
    if (live.empty()) break;
    RunScorerWavesTogether(live, scorers);
    for (CtcPrefixSearch *search : live) search->EndFrame();
  }
}

}  // namespace sherpa_onnx

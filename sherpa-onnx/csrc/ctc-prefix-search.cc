// sherpa-onnx/csrc/ctc-prefix-search.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/ctc-prefix-search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

constexpr float kLogZero = -std::numeric_limits<float>::infinity();
constexpr uint32_t kNoRecord = 0xffffffffu;
constexpr uint32_t kRootRecord = 0;

float LogAdd(float a, float b) {
  if (a == kLogZero) return b;
  if (b == kLogZero) return a;
  float high = std::max(a, b);
  float low = std::min(a, b);
  return high + std::log1p(std::exp(low - high));
}

size_t RoundUpTo8(size_t n) { return (n + 7) & ~static_cast<size_t>(7); }

uint64_t ChildKey(uint32_t parent, int32_t token) {
  return (static_cast<uint64_t>(parent) << 32) | static_cast<uint32_t>(token);
}

}  // namespace

// The fixed part of a prefix's record. Scorer subtotals and state bytes
// follow it in the same allocation.
struct CtcPrefixSearch::Record {
  uint32_t parent;
  int32_t token;  // Last token; -1 for the empty prefix.
  // Alignment mass ending in blank and in the last token, through the last
  // finished frame. Meaningful while active.
  float log_blank;
  float log_nonblank;
  // The same for the frame in progress, valid when stamp == frame + 1.
  float next_blank;
  float next_nonblank;
  int32_t stamp;
  int32_t frame;     // Frame whose emission of the last token weighed most.
  float emit_score;  // That emission's log mass.
  int32_t length;    // Tokens.
  int32_t words;     // Completed words.
  int32_t num_children;
  float scorer_total;  // Weighted sum of the scorers' subtotals.
  bool word_open;      // Text has been emitted since the last word boundary.
  bool scored;         // Scorer states and subtotals are valid.
  bool rejected;       // A scorer rejected this prefix.
  bool active;
  bool live;
};

// Scoring of one new record by one scorer: the units still to advance
// through, in units_[begin, begin + length).
struct CtcPrefixSearch::Pending {
  uint32_t record;
  uint32_t begin;
  uint32_t length;
  uint32_t position;
  float sum;
  bool rejected;
};

CtcPrefixSearch::CtcPrefixSearch(
    const CtcPrefixSearchOptions &options,
    std::shared_ptr<const CtcSearchVocabulary> vocabulary,
    std::vector<Scorer *> scorers, std::vector<float> scorer_weights)
    : options_(options),
      vocabulary_(std::move(vocabulary)),
      scorers_(std::move(scorers)),
      weights_(std::move(scorer_weights)) {
  size_t num_scorers = scorers_.size();
  subtotals_offset_ = RoundUpTo8(sizeof(Record));
  size_t offset = subtotals_offset_ + RoundUpTo8(num_scorers * sizeof(float));
  for (Scorer *scorer : scorers_) {
    state_offsets_.push_back(offset);
    offset += RoundUpTo8(scorer->Info().state_bytes);
    utterances_.push_back(scorer->NewUtterance());
  }
  record_bytes_ = offset;
  recombine_ = options_.alignment == CtcAlignmentMode::kMax;
  for (Scorer *scorer : scorers_) {
    recombine_ = recombine_ && scorer->Info().recombines;
  }
  pending_.resize(num_scorers);
  wave_.resize(num_scorers);
  scratch_.resize(num_scorers);

  uint32_t root = NewRecord(kNoRecord, -1);
  Record *record = Rec(root);
  record->log_blank = 0;
  record->active = true;
  record->scored = true;
  for (size_t s = 0; s != num_scorers; ++s) {
    scorers_[s]->Start(utterances_[s].get(), StateOf(record, s));
    SubtotalsOf(record)[s] = 0;
  }
  active_.push_back(root);
}

CtcPrefixSearch::~CtcPrefixSearch() {
  for (uint32_t index = 0; index != num_records_; ++index) {
    Record *record = Rec(index);
    if (!record->live || !record->scored) continue;
    for (size_t s = 0; s != scorers_.size(); ++s) {
      if (auto destroy = scorers_[s]->Info().destroy_state) {
        destroy(StateOf(record, s));
      }
    }
  }
}

CtcPrefixSearch::Record *CtcPrefixSearch::Rec(uint32_t index) const {
  uint64_t *block = blocks_[index >> kBlockShift].get();
  size_t slot = index & ((1u << kBlockShift) - 1);
  return reinterpret_cast<Record *>(reinterpret_cast<std::byte *>(block) +
                                    slot * record_bytes_);
}

std::byte *CtcPrefixSearch::StateOf(Record *record, int32_t scorer) const {
  return reinterpret_cast<std::byte *>(record) + state_offsets_[scorer];
}

float *CtcPrefixSearch::SubtotalsOf(Record *record) const {
  return reinterpret_cast<float *>(reinterpret_cast<std::byte *>(record) +
                                   subtotals_offset_);
}

float CtcPrefixSearch::Accumulate(float sum, float term) const {
  return options_.alignment == CtcAlignmentMode::kSum ? LogAdd(sum, term)
                                                      : std::max(sum, term);
}

float CtcPrefixSearch::Acoustic(float blank, float nonblank) const {
  return Accumulate(blank, nonblank);
}

float CtcPrefixSearch::LengthBonus(const Record &record, bool at_end) const {
  if (options_.length_bonus == 0) return 0;
  int32_t length = record.length;
  if (options_.length_bonus_unit == LengthBonusUnit::kWord) {
    length = record.words + (at_end && record.word_open ? 1 : 0);
  }
  return options_.length_bonus * length;
}

uint32_t CtcPrefixSearch::NewRecord(uint32_t parent, int32_t token) {
  uint32_t index;
  if (!free_records_.empty()) {
    index = free_records_.back();
    free_records_.pop_back();
  } else {
    index = num_records_++;
    if ((index >> kBlockShift) == blocks_.size()) {
      size_t words = (record_bytes_ / 8) << kBlockShift;
      blocks_.emplace_back(new uint64_t[words]);
    }
  }
  ++num_live_records_;

  Record *record = Rec(index);
  record->parent = parent;
  record->token = token;
  record->log_blank = kLogZero;
  record->log_nonblank = kLogZero;
  record->next_blank = kLogZero;
  record->next_nonblank = kLogZero;
  record->stamp = 0;
  record->frame = frame_;
  record->emit_score = kLogZero;
  record->length = 0;
  record->words = 0;
  record->num_children = 0;
  record->scorer_total = 0;
  record->word_open = false;
  record->scored = false;
  record->rejected = false;
  record->active = false;
  record->live = true;

  if (parent != kNoRecord) {
    Record *up = Rec(parent);
    const CtcSearchToken &info = vocabulary_->tokens[token];
    bool closes_word = info.starts_word && up->word_open;
    record->length = up->length + 1;
    record->words = up->words + (closes_word ? 1 : 0);
    record->word_open = info.starts_word
                            ? !info.chars.empty()
                            : (up->word_open || !info.chars.empty());
    up->num_children += 1;
    children_.emplace(ChildKey(parent, token), index);
  }
  return index;
}

void CtcPrefixSearch::FreeRecord(uint32_t index) {
  Record *record = Rec(index);
  if (record->scored) {
    for (size_t s = 0; s != scorers_.size(); ++s) {
      if (auto destroy = scorers_[s]->Info().destroy_state) {
        destroy(StateOf(record, s));
      }
    }
  }
  children_.erase(ChildKey(record->parent, record->token));
  Rec(record->parent)->num_children -= 1;
  record->live = false;
  free_records_.push_back(index);
  --num_live_records_;
}

// Frees a prefix that is out of the beam and has no descendants, then any
// ancestors that this leaves in the same condition.
void CtcPrefixSearch::ReleaseIfUnused(uint32_t index) {
  while (index != kRootRecord) {
    Record *record = Rec(index);
    if (!record->live || record->active || record->num_children != 0) return;
    uint32_t parent = record->parent;
    FreeRecord(index);
    index = parent;
  }
}

uint32_t CtcPrefixSearch::FindOrCreateChild(uint32_t parent, int32_t token) {
  auto it = children_.find(ChildKey(parent, token));
  if (it != children_.end()) return it->second;
  uint32_t index = NewRecord(parent, token);
  QueueScoring(index);
  return index;
}

void CtcPrefixSearch::Touch(uint32_t index) {
  Record *record = Rec(index);
  if (record->stamp == frame_ + 1) return;
  record->stamp = frame_ + 1;
  record->next_blank = kLogZero;
  record->next_nonblank = kLogZero;
  touched_.push_back(index);
}

void CtcPrefixSearch::QueueScoring(uint32_t index) {
  new_records_.push_back(index);
  Record *record = Rec(index);
  const CtcSearchToken &info = vocabulary_->tokens[record->token];
  bool closes_word = info.starts_word && Rec(record->parent)->word_open;
  for (size_t s = 0; s != scorers_.size(); ++s) {
    Pending pending = {index, static_cast<uint32_t>(units_.size()), 0, 0, 0,
                       false};
    if (scorers_[s]->Info().unit == ScorerUnit::kAsrToken) {
      units_.push_back(record->token);
    } else {
      if (closes_word) units_.push_back(options_.word_boundary_char);
      units_.insert(units_.end(), info.chars.begin(), info.chars.end());
    }
    pending.length = static_cast<uint32_t>(units_.size()) - pending.begin;
    pending_[s].push_back(pending);
  }
}

void CtcPrefixSearch::BeginFrame(const float *log_probs) {
  touched_.clear();
  new_records_.clear();
  units_.clear();
  for (auto &pending : pending_) pending.clear();

  // The most probable emittable tokens, ties to the lower ID.
  int32_t blank = vocabulary_->blank_id;
  int32_t vocab_size = static_cast<int32_t>(vocabulary_->tokens.size());
  size_t limit =
      static_cast<size_t>(std::max(options_.max_token_candidates, 0));
  candidates_.clear();
  for (int32_t token = 0; token != vocab_size; ++token) {
    if (token == blank || !vocabulary_->tokens[token].emit) continue;
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
    Record *record = Rec(index);
    float log_blank = record->log_blank;
    float log_nonblank = record->log_nonblank;
    float total = Acoustic(log_blank, log_nonblank);
    int32_t last = record->token;

    Touch(index);
    record->next_blank =
        Accumulate(record->next_blank, total + log_probs[blank]);
    if (last >= 0) {
      record->next_nonblank =
          Accumulate(record->next_nonblank, log_nonblank + log_probs[last]);
    }
    for (int32_t token : candidates_) {
      // Repeating the last token extends the prefix only after a blank.
      float source = token == last ? log_blank : total;
      if (source == kLogZero) continue;
      uint32_t child_index = FindOrCreateChild(index, token);
      Touch(child_index);
      Record *child = Rec(child_index);
      float emitted = source + log_probs[token];
      child->next_nonblank = Accumulate(child->next_nonblank, emitted);
      if (emitted > child->emit_score) {
        child->emit_score = emitted;
        child->frame = frame_;
      }
    }
  }

  for (size_t s = 0; s != scorers_.size(); ++s) {
    size_t words = RoundUpTo8(scorers_[s]->Info().state_bytes) / 8;
    scratch_[s].resize(pending_[s].size() * 2 * words);
  }
}

bool CtcPrefixSearch::GatherScorerWave(
    int32_t scorer, std::vector<ScorerExtension> *extensions) {
  auto &wave = wave_[scorer];
  wave.clear();
  size_t state_words = RoundUpTo8(scorers_[scorer]->Info().state_bytes) / 8;
  auto &pending_list = pending_[scorer];
  for (size_t i = 0; i != pending_list.size(); ++i) {
    Pending &pending = pending_list[i];
    if (pending.rejected || pending.position == pending.length) continue;
    Record *record = Rec(pending.record);
    // A token of several units advances through intermediate states kept in
    // two alternating scratch buffers.
    auto scratch = [&](uint32_t position) {
      return reinterpret_cast<std::byte *>(
          scratch_[scorer].data() + (i * 2 + (position & 1)) * state_words);
    };
    ScorerExtension extension;
    extension.parent.utterance = utterances_[scorer].get();
    extension.parent.bytes = pending.position == 0
                                 ? StateOf(Rec(record->parent), scorer)
                                 : scratch(pending.position - 1);
    extension.unit = units_[pending.begin + pending.position];
    extension.child = pending.position + 1 == pending.length
                          ? StateOf(record, scorer)
                          : scratch(pending.position);
    extensions->push_back(extension);
    wave.push_back(static_cast<uint32_t>(i));
  }
  return !wave.empty();
}

void CtcPrefixSearch::ApplyScorerWave(int32_t scorer,
                                      const ScorerAdvanceResult *results) {
  size_t state_words = RoundUpTo8(scorers_[scorer]->Info().state_bytes) / 8;
  auto destroy = scorers_[scorer]->Info().destroy_state;
  const auto &wave = wave_[scorer];
  for (size_t j = 0; j != wave.size(); ++j) {
    size_t i = wave[j];
    Pending &pending = pending_[scorer][i];
    if (pending.position > 0 && destroy != nullptr) {
      destroy(reinterpret_cast<std::byte *>(
          scratch_[scorer].data() +
          (i * 2 + ((pending.position - 1) & 1)) * state_words));
    }
    if (results[j].rejected) {
      pending.rejected = true;
    } else {
      pending.sum += results[j].log_score;
      pending.position += 1;
    }
  }
}

// Completes the records created this frame from their scorers' results.
void CtcPrefixSearch::FinishScoring() {
  size_t num_scorers = scorers_.size();
  for (size_t n = 0; n != new_records_.size(); ++n) {
    Record *record = Rec(new_records_[n]);
    Record *parent = Rec(record->parent);
    bool rejected = false;
    for (size_t s = 0; s != num_scorers; ++s) {
      rejected = rejected || pending_[s][n].rejected;
    }
    if (rejected) {
      // Discard the states the other scorers did write.
      for (size_t s = 0; s != num_scorers; ++s) {
        const Pending &pending = pending_[s][n];
        auto destroy = scorers_[s]->Info().destroy_state;
        if (!pending.rejected && pending.length != 0 && destroy != nullptr) {
          destroy(StateOf(record, s));
        }
      }
      record->rejected = true;
      continue;
    }
    float total = parent->scorer_total;
    for (size_t s = 0; s != num_scorers; ++s) {
      const Pending &pending = pending_[s][n];
      const ScorerInfo &info = scorers_[s]->Info();
      if (pending.length == 0) {
        // No unit for this scorer: the state is the parent's.
        if (info.copy_state != nullptr) {
          info.copy_state(StateOf(parent, s), StateOf(record, s));
        } else {
          std::memcpy(StateOf(record, s), StateOf(parent, s), info.state_bytes);
        }
      }
      SubtotalsOf(record)[s] = SubtotalsOf(parent)[s] + pending.sum;
      total += weights_[s] * pending.sum;
    }
    record->scorer_total = total;
    record->scored = true;
  }
}

bool CtcPrefixSearch::SameFuture(const Record &a, const Record &b) const {
  if (a.token != b.token || a.word_open != b.word_open) return false;
  for (size_t s = 0; s != scorers_.size(); ++s) {
    if (std::memcmp(StateOf(const_cast<Record *>(&a), s),
                    StateOf(const_cast<Record *>(&b), s),
                    scorers_[s]->Info().state_bytes) != 0) {
      return false;
    }
  }
  return true;
}

// Among this frame's hypotheses with the same future (last token, word
// state, and scorer states), drops each one that another matches or beats
// both ending in blank and ending in its last token. Everything scored
// from here on is the same for both, so the dropped one cannot win.
void CtcPrefixSearch::Recombine() {
  same_hash_.clear();
  for (size_t i = 0; i != touched_.size(); ++i) {
    if (touched_scores_[i] == kLogZero) continue;
    Record *record = Rec(touched_[i]);
    uint64_t hash = 1469598103934665603ull;
    auto mix = [&hash](const void *data, size_t size) {
      const auto *bytes = static_cast<const unsigned char *>(data);
      for (size_t k = 0; k != size; ++k) {
        hash = (hash ^ bytes[k]) * 1099511628211ull;
      }
    };
    mix(&record->token, sizeof(record->token));
    mix(&record->word_open, sizeof(record->word_open));
    for (size_t s = 0; s != scorers_.size(); ++s) {
      mix(StateOf(record, s), scorers_[s]->Info().state_bytes);
    }
    same_hash_[hash].push_back(static_cast<uint32_t>(i));
  }
  for (auto &entry : same_hash_) {
    std::vector<uint32_t> &group = entry.second;
    for (size_t x = 0; x != group.size(); ++x) {
      if (touched_scores_[group[x]] == kLogZero) continue;
      const Record &a = *Rec(touched_[group[x]]);
      float a_rest = a.scorer_total + LengthBonus(a, false);
      for (size_t y = 0; y != group.size(); ++y) {
        if (x == y || touched_scores_[group[y]] == kLogZero) continue;
        const Record &b = *Rec(touched_[group[y]]);
        if (!SameFuture(a, b)) continue;
        float b_rest = b.scorer_total + LengthBonus(b, false);
        if (a.next_blank + a_rest >= b.next_blank + b_rest &&
            a.next_nonblank + a_rest >= b.next_nonblank + b_rest) {
          touched_scores_[group[y]] = kLogZero;
          ++num_recombined_;
        }
      }
    }
  }
}

void CtcPrefixSearch::EndFrame() {
  FinishScoring();

  touched_scores_.resize(touched_.size());
  for (size_t i = 0; i != touched_.size(); ++i) {
    const Record &record = *Rec(touched_[i]);
    touched_scores_[i] =
        record.rejected ? kLogZero
                        : Acoustic(record.next_blank, record.next_nonblank) +
                              record.scorer_total + LengthBonus(record, false);
  }

  if (recombine_) Recombine();

  // Best first; equal scores keep the order of their records' indices.
  std::vector<uint32_t> order(touched_.size());
  std::iota(order.begin(), order.end(), 0);
  size_t keep = std::min(order.size(),
                         static_cast<size_t>(std::max(options_.max_active, 0)));
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

  for (uint32_t index : active_) Rec(index)->active = false;
  active_.clear();
  for (size_t i = 0; i != keep; ++i) {
    uint32_t index = touched_[order[i]];
    Record *record = Rec(index);
    record->log_blank = record->next_blank;
    record->log_nonblank = record->next_nonblank;
    record->active = true;
    active_.push_back(index);
  }
  for (size_t i = keep; i != order.size(); ++i) {
    ReleaseIfUnused(touched_[order[i]]);
  }
  num_kept_ += static_cast<int64_t>(keep);
  num_scored_ += static_cast<int64_t>(new_records_.size());
  ++frame_;
}

void CtcPrefixSearch::Decode(const float *log_probs, int32_t num_frames) {
  size_t vocab_size = vocabulary_->tokens.size();
  std::vector<ScorerExtension> extensions;
  std::vector<ScorerAdvanceResult> results;
  for (int32_t t = 0; t != num_frames; ++t) {
    BeginFrame(log_probs + t * vocab_size);
    for (size_t s = 0; s != scorers_.size(); ++s) {
      extensions.clear();
      while (GatherScorerWave(s, &extensions)) {
        results.assign(extensions.size(), ScorerAdvanceResult());
        scorers_[s]->Advance({extensions.data(), extensions.size()},
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
  Record *record = Rec(index);
  for (size_t s = 0; s != scorers_.size(); ++s) {
    hypothesis.scorer_log_scores.push_back(
        SubtotalsOf(record)[s] + (final_scores ? final_scores[s] : 0));
  }
  for (; record->parent != kNoRecord; record = Rec(record->parent)) {
    hypothesis.tokens.push_back(record->token);
    hypothesis.timestamps.push_back(record->frame);
  }
  std::reverse(hypothesis.tokens.begin(), hypothesis.tokens.end());
  std::reverse(hypothesis.timestamps.begin(), hypothesis.timestamps.end());
  return hypothesis;
}

CtcPrefixHypothesis CtcPrefixSearch::Best() const {
  uint32_t best = kNoRecord;
  float best_score = kLogZero;
  float best_acoustic = kLogZero;
  for (uint32_t index : active_) {
    const Record &record = *Rec(index);
    float acoustic = Acoustic(record.log_blank, record.log_nonblank);
    float score = acoustic + record.scorer_total + LengthBonus(record, false);
    if (best == kNoRecord || score > best_score) {
      best = index;
      best_score = score;
      best_acoustic = acoustic;
    }
  }
  if (best == kNoRecord) return CtcPrefixHypothesis();
  return MakeHypothesis(best, best_score, best_acoustic, nullptr);
}

std::vector<CtcPrefixHypothesis> CtcPrefixSearch::Finish(int32_t nbest) {
  size_t num_scorers = scorers_.size();
  size_t num_active = active_.size();
  // final_scores[i * num_scorers + s]
  std::vector<float> final_scores(num_active * num_scorers, 0);
  std::vector<bool> accepted(num_active, true);
  std::vector<ScorerStateRef> states(num_active);
  std::vector<ScorerFinalScore> results(num_active);
  for (size_t s = 0; s != num_scorers; ++s) {
    for (size_t i = 0; i != num_active; ++i) {
      states[i] = {utterances_[s].get(), StateOf(Rec(active_[i]), s)};
    }
    scorers_[s]->Finish({states.data(), num_active}, results.data());
    for (size_t i = 0; i != num_active; ++i) {
      final_scores[i * num_scorers + s] = results[i].log_score;
      if (!results[i].accepted) accepted[i] = false;
    }
  }

  std::vector<std::pair<float, size_t>> ranked;
  for (size_t i = 0; i != num_active; ++i) {
    if (!accepted[i]) continue;
    const Record &record = *Rec(active_[i]);
    float score = Acoustic(record.log_blank, record.log_nonblank) +
                  record.scorer_total + LengthBonus(record, true);
    for (size_t s = 0; s != num_scorers; ++s) {
      score += weights_[s] * final_scores[i * num_scorers + s];
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
    const Record &record = *Rec(active_[entry.second]);
    hypotheses.push_back(
        MakeHypothesis(active_[entry.second], entry.first,
                       Acoustic(record.log_blank, record.log_nonblank),
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

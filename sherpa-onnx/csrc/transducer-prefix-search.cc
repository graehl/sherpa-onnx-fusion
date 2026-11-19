// sherpa-onnx/csrc/transducer-prefix-search.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/transducer-prefix-search.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

}  // namespace

// The search's state for one prefix: where its probability sits in time.
struct TransducerPrefixSearch::State {
  float now;    // At the current frame, to expand in this round.
  float again;  // At the current frame, for the next round.
  // At later frames, by frame modulo the ring size.
  float arrive[kMaxDuration + 1];
  int32_t pins;      // Lists that hold this prefix.
  int32_t frame;     // Frame whose emission of the last token weighed most.
  float emit_score;  // That emission's log mass.
};

// What the probability in one queue needed: on every path that added to it,
// and on the path that added the most ("dominant"), whose term is kept.
struct TransducerPrefixSearch::QueueNeed {
  TransducerSurvivalNeed all;
  TransducerSurvivalNeed dominant;
  float dominant_term = kLogZero;

  // The first contribution of "term" from a queue whose need is "from".
  void Start(const QueueNeed &from, float term) {
    all = from.all;
    dominant = from.dominant;
    dominant_term = term;
  }
  void Add(const QueueNeed &from, float term) {
    all.Absorb(from.all);
    if (term > dominant_term) {
      dominant = from.dominant;
      dominant_term = term;
    }
  }
  // What being kept at "rank" and "gap" adds; "wait" for a waiting list.
  void Stamp(int32_t rank, float gap, bool wait) {
    TransducerSurvivalNeed here;
    (wait ? here.wait_rank : here.rank) = rank;
    (wait ? here.wait_gap : here.gap) = gap;
    all.Absorb(here);
    dominant.Absorb(here);
  }
};

// With trace_survival, after the State: what the probability in each of the
// prefix's queues needed, matching State's fields.
struct TransducerPrefixSearch::Trace {
  QueueNeed now;
  QueueNeed again;
  QueueNeed arrive[kMaxDuration + 1];
};

const size_t TransducerPrefixSearch::kTraceOffset = (sizeof(State) + 7) / 8 * 8;

void TransducerSurvivalNeed::Absorb(const TransducerSurvivalNeed &other) {
  gap = std::max(gap, other.gap);
  rank = std::max(rank, other.rank);
  wait_gap = std::max(wait_gap, other.wait_gap);
  wait_rank = std::max(wait_rank, other.wait_rank);
}

void TransducerWaitHistogram::Add(int32_t rank, float gap) {
  int32_t r = 0;
  while (r + 1 < kNumRanks && rank > kRanks[r]) ++r;
  int32_t g = 0;
  while (g + 1 < kNumGaps && gap > kGaps[g]) ++g;
  counts[static_cast<size_t>(r) * kNumGaps + g] += 1;
}

void TransducerWaitHistogram::AddLength(int32_t offset,
                                        std::vector<int64_t> *lengths) {
  offset = std::min(std::max(offset, -kMaxOffset), kMaxOffset);
  (*lengths)[offset + kMaxOffset] += 1;
}

void TransducerWaitHistogram::Merge(const TransducerWaitHistogram &other) {
  for (size_t i = 0; i != counts.size(); ++i) counts[i] += other.counts[i];
  for (size_t i = 0; i != waiting_lengths.size(); ++i) {
    waiting_lengths[i] += other.waiting_lengths[i];
    kept_lengths[i] += other.kept_lengths[i];
    dropped_lengths[i] += other.dropped_lengths[i];
  }
  rounds += other.rounds;
  peak_waiting = std::max(peak_waiting, other.peak_waiting);
}

TransducerPrefixSearch::TransducerPrefixSearch(
    const TransducerPrefixSearchOptions &options,
    std::shared_ptr<const SearchVocabulary> vocabulary,
    std::vector<Scorer *> scorers, std::vector<float> scorer_weights)
    : options_(options),
      trie_(
          std::move(vocabulary), std::move(scorers), std::move(scorer_weights),
          options.trace_survival ? kTraceOffset + sizeof(Trace) : sizeof(State),
          options.word_boundary_char) {
  if (options_.topology == TransducerTopology::kTdt) {
    options_.max_duration =
        std::min(std::max(options_.max_duration, 1), kMaxDuration);
    ring_size_ = options_.max_duration + 1;
  }
  arriving_.resize(ring_size_);

  Node *root = trie_.At(ScoredPrefixTrie::kRoot);
  State *state = StateOf(root);
  state->now = 0;
  state->again = kLogZero;
  std::fill(std::begin(state->arrive), std::end(state->arrive), kLogZero);
  state->pins = 0;
  state->frame = 0;
  state->emit_score = kLogZero;
  if (options_.trace_survival) {
    *TraceOf(root) = Trace();
    TraceOf(root)->now.Stamp(1, 0, false);
  }
  Pin(ScoredPrefixTrie::kRoot);
  frontier_.push_back(ScoredPrefixTrie::kRoot);
}

TransducerPrefixSearch::State *TransducerPrefixSearch::StateOf(
    Node *node) const {
  return reinterpret_cast<State *>(trie_.SearchState(node));
}

TransducerPrefixSearch::Trace *TransducerPrefixSearch::TraceOf(
    Node *node) const {
  return reinterpret_cast<Trace *>(trie_.SearchState(node) + kTraceOffset);
}

void TransducerPrefixSearch::NeedOf(Node *node, TransducerSurvivalNeed *all,
                                    TransducerSurvivalNeed *dominant) const {
  *all = TransducerSurvivalNeed();
  *dominant = TransducerSurvivalNeed();
  if (!options_.trace_survival) return;
  const State &state = *StateOf(node);
  const Trace &trace = *TraceOf(node);
  float largest = kLogZero;
  auto add = [&](float mass, const QueueNeed &need) {
    if (mass == kLogZero) return;
    all->Absorb(need.all);
    if (mass > largest) {
      largest = mass;
      *dominant = need.dominant;
    }
  };
  add(state.now, trace.now);
  add(state.again, trace.again);
  for (int32_t k = 0; k != ring_size_; ++k)
    add(state.arrive[k], trace.arrive[k]);
}

void TransducerPrefixSearch::RankWaiting() {
  int64_t waiting = 0;
  std::vector<std::pair<float, uint32_t>> ranked;
  for (int32_t slot = 0; slot != ring_size_; ++slot) {
    ranked.clear();
    for (uint32_t index : arriving_[slot]) {
      Node *node = trie_.At(index);
      if (node->rejected) continue;  // Never survives its frame.
      ranked.emplace_back(StateOf(node)->arrive[slot] + node->scorer_total +
                              LengthBonus(*node, false),
                          index);
    }
    // As Prune orders: score, then index.
    std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
      if (a.first != b.first) return a.first > b.first;
      return a.second < b.second;
    });
    for (size_t i = 0; i != ranked.size(); ++i) {
      int32_t rank = static_cast<int32_t>(i) + 1;
      float gap = ranked[0].first - ranked[i].first;
      TraceOf(trie_.At(ranked[i].second))->arrive[slot].Stamp(rank, gap, true);
      waiting_.Add(rank, gap);
      TransducerWaitHistogram::AddLength(trie_.At(ranked[i].second)->length -
                                             trie_.At(ranked[0].second)->length,
                                         &waiting_.waiting_lengths);
    }
    waiting += static_cast<int64_t>(ranked.size());
  }
  waiting_.rounds += 1;
  waiting_.peak_waiting = std::max(waiting_.peak_waiting, waiting);
}

void TransducerPrefixSearch::Pin(uint32_t index) {
  Node *node = trie_.At(index);
  if (StateOf(node)->pins++ == 0) node->pinned = true;
}

void TransducerPrefixSearch::Unpin(uint32_t index) {
  Node *node = trie_.At(index);
  if (--StateOf(node)->pins == 0) {
    node->pinned = false;
    trie_.ReleaseIfUnused(index);
  }
}

void TransducerPrefixSearch::AddAgain(uint32_t index, float log_mass,
                                      const QueueNeed *need) {
  if (log_mass == kLogZero) return;
  Node *node = trie_.At(index);
  State *state = StateOf(node);
  if (state->again == kLogZero) {
    again_.push_back(index);
    Pin(index);
    if (need != nullptr) TraceOf(node)->again.Start(*need, log_mass);
  } else if (need != nullptr) {
    TraceOf(node)->again.Add(*need, log_mass);
  }
  state->again = LogAdd(state->again, log_mass);
}

void TransducerPrefixSearch::AddArrival(uint32_t index, int32_t frames_ahead,
                                        float log_mass, const QueueNeed *need) {
  if (log_mass == kLogZero) return;
  int32_t slot = (frame_ + frames_ahead) % ring_size_;
  Node *node = trie_.At(index);
  State *state = StateOf(node);
  if (state->arrive[slot] == kLogZero) {
    arriving_[slot].push_back(index);
    Pin(index);
    if (need != nullptr) TraceOf(node)->arrive[slot].Start(*need, log_mass);
  } else if (need != nullptr) {
    TraceOf(node)->arrive[slot].Add(*need, log_mass);
  }
  state->arrive[slot] = LogAdd(state->arrive[slot], log_mass);
}

float TransducerPrefixSearch::LengthBonus(const Node &node, bool at_end) const {
  if (options_.length_bonus == 0) return 0;
  return options_.length_bonus *
         trie_.LengthIn(node, options_.length_bonus_unit, at_end);
}

void TransducerPrefixSearch::ContextOf(uint32_t hypothesis, int32_t size,
                                       int64_t pad, int64_t *context) const {
  const Node *node = trie_.At(hypothesis);
  for (int32_t i = size - 1; i >= 0; --i) {
    if (node->parent == ScoredPrefixTrie::kNone) {
      context[i] = pad;
    } else {
      context[i] = node->token;
      node = trie_.At(node->parent);
    }
  }
}

int32_t TransducerPrefixSearch::LastTokenOf(uint32_t hypothesis) const {
  return trie_.At(hypothesis)->token;
}

int32_t TransducerPrefixSearch::LengthOf(uint32_t hypothesis) const {
  return trie_.At(hypothesis)->length;
}

uint32_t TransducerPrefixSearch::ParentOf(uint32_t hypothesis) const {
  return trie_.At(hypothesis)->parent;
}

uint32_t TransducerPrefixSearch::SerialOf(uint32_t hypothesis) const {
  return trie_.At(hypothesis)->serial;
}

void TransducerPrefixSearch::Expand(const float *log_probs,
                                    const float *duration_log_probs) {
  trie_.StartScoring();
  const SearchVocabulary &vocabulary = trie_.Vocabulary();
  int32_t blank = vocabulary.blank_id;
  int32_t vocab_size = static_cast<int32_t>(vocabulary.tokens.size());
  size_t limit =
      static_cast<size_t>(std::max(options_.max_token_candidates, 0));
  bool tdt = options_.topology == TransducerTopology::kTdt;
  bool tokens_stay = options_.topology == TransducerTopology::kRnnt;
  // In the last round of a frame nothing may stay in the frame.
  bool last_round = round_ + 1 >= options_.max_symbols_per_frame;
  int32_t num_durations = options_.max_duration + 1;

  bool prune_steps =
      options_.prune_steps &&
      options_.topology == TransducerTopology::kOneSymbolPerFrame;
  if (prune_steps) {
    // The best max_active steps over the whole frontier, best first.
    size_t keep = static_cast<size_t>(std::max(options_.max_active, 0));
    steps_.clear();
    for (size_t i = 0; i != frontier_.size(); ++i) {
      const Node &node = *trie_.At(frontier_[i]);
      const float *row = log_probs + i * vocab_size;
      float base = StateOf(trie_.At(frontier_[i]))->now + node.scorer_total +
                   LengthBonus(node, false);
      for (int32_t token = 0; token != vocab_size; ++token) {
        if (token != blank && token != options_.blank_like_token &&
            !vocabulary.tokens[token].emit) {
          continue;
        }
        float key = base + row[token];
        if (steps_.size() == keep &&
            (keep == 0 || !(key > steps_.back().key))) {
          continue;
        }
        auto position = std::upper_bound(
            steps_.begin(), steps_.end(), key,
            [](float k, const Step &step) { return k > step.key; });
        steps_.insert(position, {key, static_cast<uint32_t>(i), token});
        if (steps_.size() > keep) steps_.pop_back();
      }
    }
  }

  for (size_t i = 0; i != frontier_.size(); ++i) {
    uint32_t index = frontier_[i];
    const float *row = log_probs + i * vocab_size;
    const float *durations =
        tdt ? duration_log_probs + i * num_durations : nullptr;
    float mass = StateOf(trie_.At(index))->now;
    // Records are never moved, so this stays valid while children are made.
    const QueueNeed *need =
        options_.trace_survival ? &TraceOf(trie_.At(index))->now : nullptr;

    // Blank: the same prefix at a later frame. A blank with no advance is
    // treated as advancing one frame.
    if (prune_steps) {
      for (const Step &step : steps_) {
        if (step.position != i) continue;
        if (step.token == blank || step.token == options_.blank_like_token) {
          AddArrival(index, 1, mass + row[step.token], need);
        }
      }
    } else if (tdt) {
      for (int32_t d = 0; d != num_durations; ++d) {
        AddArrival(index, std::max(d, 1), mass + row[blank] + durations[d],
                   need);
      }
    } else {
      AddArrival(index, 1, mass + row[blank], need);
    }

    // The most probable emittable tokens, ties to the lower ID.
    candidates_.clear();
    if (prune_steps) {
      for (const Step &step : steps_) {
        if (step.position == i && step.token != blank &&
            step.token != options_.blank_like_token) {
          candidates_.push_back(step.token);
        }
      }
    }
    for (int32_t token = 0; !prune_steps && token != vocab_size; ++token) {
      if (token == blank || !vocabulary.tokens[token].emit) continue;
      if (candidates_.size() == limit &&
          (limit == 0 || !(row[token] > row[candidates_.back()]))) {
        continue;
      }
      auto position = std::upper_bound(
          candidates_.begin(), candidates_.end(), token,
          [row](int32_t a, int32_t b) { return row[a] > row[b]; });
      candidates_.insert(position, token);
      if (candidates_.size() > limit) candidates_.pop_back();
    }

    // In a frame's last round an RNN-T token would have to stay: none.
    if (tokens_stay && last_round) candidates_.clear();

    for (int32_t token : candidates_) {
      bool created = false;
      uint32_t child_index = trie_.FindOrCreateChild(index, token, &created);
      State *child = StateOf(trie_.At(child_index));
      if (created) {
        child->now = kLogZero;
        child->again = kLogZero;
        std::fill(std::begin(child->arrive), std::end(child->arrive), kLogZero);
        child->pins = 0;
        child->frame = frame_;
        child->emit_score = kLogZero;
        if (need != nullptr) *TraceOf(trie_.At(child_index)) = Trace();
      }
      float emitted = mass + row[token];
      float largest = kLogZero;
      if (tdt) {
        for (int32_t d = 0; d != num_durations; ++d) {
          float term = emitted + durations[d];
          if (d == 0 && last_round) continue;
          if (d == 0) {
            AddAgain(child_index, term, need);
          } else {
            AddArrival(child_index, d, term, need);
          }
          largest = std::max(largest, term);
        }
      } else if (tokens_stay) {
        AddAgain(child_index, emitted, need);
        largest = emitted;
      } else {
        AddArrival(child_index, 1, emitted, need);
        largest = emitted;
      }
      if (largest > child->emit_score) {
        child->emit_score = largest;
        child->frame = frame_;
      }
    }
  }

  // The frontier's probability has all moved on.
  for (uint32_t index : frontier_) {
    StateOf(trie_.At(index))->now = kLogZero;
  }
}

bool TransducerPrefixSearch::GatherScorerWave(
    int32_t scorer, std::vector<ScorerExtension> *extensions) {
  return trie_.GatherScorerWave(scorer, extensions);
}

void TransducerPrefixSearch::ApplyScorerWave(
    int32_t scorer, const ScorerAdvanceResult *results) {
  trie_.ApplyScorerWave(scorer, results);
}

std::vector<uint32_t> TransducerPrefixSearch::Prune(
    const std::vector<uint32_t> &candidates, int32_t slot, bool to_frontier) {
  // A waiting list is cut by its own limits, by default twice the
  // frontier's count within half its beam.
  int32_t max_active = options_.max_active;
  float beam = options_.beam;
  if (!to_frontier) {
    max_active = options_.waiting_max_active > 0 ? options_.waiting_max_active
                                                 : 2 * max_active;
    beam = options_.waiting_beam > 0 ? options_.waiting_beam : beam / 2;
  }
  auto mass_in = [slot](State *state) -> float & {
    return slot == kAgainSlot ? state->again : state->arrive[slot];
  };
  std::vector<float> scores(candidates.size());
  for (size_t i = 0; i != candidates.size(); ++i) {
    Node *node = trie_.At(candidates[i]);
    scores[i] = node->rejected ? kLogZero
                               : mass_in(StateOf(node)) + node->scorer_total +
                                     LengthBonus(*node, false);
  }
  std::vector<uint32_t> order(candidates.size());
  std::iota(order.begin(), order.end(), 0);
  size_t keep =
      std::min(order.size(), static_cast<size_t>(std::max(max_active, 0)));
  std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                    [&](uint32_t a, uint32_t b) {
                      if (scores[a] != scores[b]) return scores[a] > scores[b];
                      return candidates[a] < candidates[b];
                    });
  if (keep != 0) {
    float cutoff = scores[order[0]] - beam;
    while (keep != 0 && (scores[order[keep - 1]] == kLogZero ||
                         scores[order[keep - 1]] < cutoff)) {
      --keep;
    }
  }
  if (options_.trace_survival && to_frontier && !order.empty()) {
    // What each kept one needed, now that it moves to the frontier.
    for (size_t i = 0; i != keep; ++i) {
      Trace *trace = TraceOf(trie_.At(candidates[order[i]]));
      QueueNeed need = slot == kAgainSlot ? trace->again : trace->arrive[slot];
      need.Stamp(static_cast<int32_t>(i) + 1,
                 scores[order[0]] - scores[order[i]], false);
      trace->now = need;
    }
    if (slot != kAgainSlot) {
      int32_t best = trie_.At(candidates[order[0]])->length;
      for (size_t i = 0; i != order.size(); ++i) {
        TransducerWaitHistogram::AddLength(
            trie_.At(candidates[order[i]])->length - best,
            i < keep ? &waiting_.kept_lengths : &waiting_.dropped_lengths);
      }
    }
  }
  std::vector<uint32_t> kept;
  for (size_t i = 0; i != keep; ++i) kept.push_back(candidates[order[i]]);
  for (size_t i = keep; i != order.size(); ++i) {
    uint32_t index = candidates[order[i]];
    mass_in(StateOf(trie_.At(index))) = kLogZero;
    Unpin(index);
  }
  return kept;
}

bool TransducerPrefixSearch::EndRound() {
  trie_.FinishScoring();
  // A new prefix with no positive-probability transition joined no queue.
  for (uint32_t index : trie_.NewNodes()) trie_.ReleaseIfUnused(index);
  // The round's scores are complete: where a policy would cut waiting lists.
  if (options_.trace_survival) RankWaiting();

  // The frontier is spent; it leaves through Unpin below, after the
  // next frontier is chosen, so a prefix in both is not freed in between.
  std::vector<uint32_t> spent = std::move(frontier_);
  frontier_ = Prune(again_, kAgainSlot);
  for (uint32_t index : frontier_) {
    State *state = StateOf(trie_.At(index));
    state->now = state->again;
    state->again = kLogZero;
  }
  again_.clear();
  if (options_.prune_waiting && !options_.prune_steps) {
    // The next frame's list is left whole for EndFrame's pruning: cut while
    // its frame still has rounds, it would lose the mass of entries that
    // gain more later. So only TDT, with lists further ahead, is cut.
    const int32_t next = (frame_ + 1) % ring_size_;
    for (int32_t slot = 0; slot != ring_size_; ++slot) {
      if (slot != next) arriving_[slot] = Prune(arriving_[slot], slot, false);
    }
  }
  for (uint32_t index : spent) Unpin(index);
  ++round_;
  return !frontier_.empty();
}

void TransducerPrefixSearch::EndFrame() {
  // Anything still waiting in the frame was not expanded: drop it.
  for (uint32_t index : frontier_) {
    StateOf(trie_.At(index))->now = kLogZero;
    Unpin(index);
  }
  frontier_.clear();

  ++frame_;
  round_ = 0;
  int32_t slot = frame_ % ring_size_;
  std::vector<uint32_t> arrived = std::move(arriving_[slot]);
  arriving_[slot].clear();
  frontier_ = Prune(arrived, slot);
  for (uint32_t index : frontier_) {
    State *state = StateOf(trie_.At(index));
    state->now = state->arrive[slot];
    state->arrive[slot] = kLogZero;
  }
}

void TransducerPrefixSearch::Outstanding(std::vector<uint32_t> *nodes,
                                         std::vector<float> *log_mass) const {
  nodes->clear();
  auto add = [&](uint32_t index) {
    if (std::find(nodes->begin(), nodes->end(), index) == nodes->end()) {
      nodes->push_back(index);
    }
  };
  for (uint32_t index : frontier_) add(index);
  for (uint32_t index : again_) add(index);
  for (const auto &slot : arriving_) {
    for (uint32_t index : slot) add(index);
  }
  log_mass->clear();
  for (uint32_t index : *nodes) {
    const State &state = *StateOf(trie_.At(index));
    float total = LogAdd(state.now, state.again);
    for (int32_t k = 0; k != ring_size_; ++k) {
      total = LogAdd(total, state.arrive[k]);
    }
    log_mass->push_back(total);
  }
}

TransducerPrefixHypothesis TransducerPrefixSearch::MakeHypothesis(
    uint32_t index, float score, float log_mass,
    const float *final_scores) const {
  TransducerPrefixHypothesis hypothesis;
  hypothesis.score = score;
  hypothesis.model_log_prob = log_mass;
  Node *node = trie_.At(index);
  for (size_t s = 0; s != trie_.NumScorers(); ++s) {
    hypothesis.scorer_log_scores.push_back(
        trie_.Subtotals(node)[s] + (final_scores ? final_scores[s] : 0));
  }
  NeedOf(node, &hypothesis.required, &hypothesis.required_dominant);
  trie_.TokensOf(index, &hypothesis.tokens);
  for (; node->parent != ScoredPrefixTrie::kNone;
       node = trie_.At(node->parent)) {
    hypothesis.timestamps.push_back(StateOf(node)->frame);
  }
  std::reverse(hypothesis.timestamps.begin(), hypothesis.timestamps.end());
  return hypothesis;
}

float TransducerPrefixSearch::RankKey(const Node &node, float score) const {
  if (options_.rank_length_offset < 0) return score;
  return score / std::max(options_.rank_length_offset + node.length, 1);
}

TransducerPrefixHypothesis TransducerPrefixSearch::Best() const {
  std::vector<uint32_t> nodes;
  std::vector<float> log_mass;
  Outstanding(&nodes, &log_mass);
  size_t best = nodes.size();
  float best_score = kLogZero;
  float best_key = kLogZero;
  for (size_t i = 0; i != nodes.size(); ++i) {
    const Node &node = *trie_.At(nodes[i]);
    if (node.rejected) continue;
    float score = log_mass[i] + node.scorer_total + LengthBonus(node, false);
    float key = RankKey(node, score);
    if (best == nodes.size() || key > best_key) {
      best = i;
      best_key = key;
      best_score = score;
    }
  }
  if (best == nodes.size()) return TransducerPrefixHypothesis();
  return MakeHypothesis(nodes[best], best_score, log_mass[best], nullptr);
}

void TransducerPrefixSearch::Finishable(std::vector<uint32_t> *scored,
                                        std::vector<float> *scored_mass) const {
  std::vector<uint32_t> nodes;
  std::vector<float> log_mass;
  Outstanding(&nodes, &log_mass);
  // A rejected prefix has no scorer states to finish.
  scored->clear();
  scored_mass->clear();
  for (size_t i = 0; i != nodes.size(); ++i) {
    if (trie_.At(nodes[i])->rejected) continue;
    scored->push_back(nodes[i]);
    scored_mass->push_back(log_mass[i]);
  }
}

std::vector<TransducerPrefixHypothesis> TransducerPrefixSearch::Finish(
    int32_t nbest) {
  return std::move(FinishTogether({this}, nbest)[0]);
}

std::vector<std::vector<TransducerPrefixHypothesis>>
TransducerPrefixSearch::FinishTogether(
    const std::vector<TransducerPrefixSearch *> &searches, int32_t nbest) {
  std::vector<const ScoredPrefixTrie *> tries;
  std::vector<std::vector<uint32_t>> scored(searches.size());
  std::vector<std::vector<float>> scored_mass(searches.size());
  for (size_t i = 0; i != searches.size(); ++i) {
    tries.push_back(&searches[i]->trie_);
    searches[i]->Finishable(&scored[i], &scored_mass[i]);
  }
  std::vector<std::vector<float>> final_scores;
  std::vector<std::vector<bool>> accepted;
  ScoredPrefixTrie::FinalScoresTogether(tries, scored, &final_scores,
                                        &accepted);
  std::vector<std::vector<TransducerPrefixHypothesis>> results;
  for (size_t i = 0; i != searches.size(); ++i) {
    results.push_back(searches[i]->RankFinished(
        scored[i], scored_mass[i], final_scores[i], accepted[i], nbest));
  }
  return results;
}

std::vector<TransducerPrefixHypothesis> TransducerPrefixSearch::RankFinished(
    const std::vector<uint32_t> &scored, const std::vector<float> &scored_mass,
    const std::vector<float> &final_scores, const std::vector<bool> &accepted,
    int32_t nbest) const {
  size_t num_scorers = trie_.NumScorers();
  std::vector<std::pair<float, size_t>> ranked;
  for (size_t i = 0; i != scored.size(); ++i) {
    if (!accepted[i]) continue;
    const Node &node = *trie_.At(scored[i]);
    float score = scored_mass[i] + node.scorer_total + LengthBonus(node, true);
    for (size_t s = 0; s != num_scorers; ++s) {
      if (trie_.Weights()[s] != 0) {
        score += trie_.Weights()[s] * final_scores[i * num_scorers + s];
      }
    }
    ranked.emplace_back(score, i);
  }
  std::stable_sort(ranked.begin(), ranked.end(),
                   [this, &scored](const auto &a, const auto &b) {
                     return RankKey(*trie_.At(scored[a.second]), a.first) >
                            RankKey(*trie_.At(scored[b.second]), b.first);
                   });
  if (ranked.size() > static_cast<size_t>(std::max(nbest, 0))) {
    ranked.resize(std::max(nbest, 0));
  }

  std::vector<TransducerPrefixHypothesis> hypotheses;
  for (const auto &entry : ranked) {
    hypotheses.push_back(MakeHypothesis(
        scored[entry.second], entry.first, scored_mass[entry.second],
        final_scores.data() + entry.second * num_scorers));
  }
  return hypotheses;
}

void RunScorerWaves(TransducerPrefixSearch *search,
                    const std::vector<Scorer *> &scorers) {
  std::vector<ScorerExtension> extensions;
  std::vector<ScorerAdvanceResult> results;
  for (size_t s = 0; s != scorers.size(); ++s) {
    extensions.clear();
    while (search->GatherScorerWave(s, &extensions)) {
      results.assign(extensions.size(), ScorerAdvanceResult());
      scorers[s]->Advance({extensions.data(), extensions.size()},
                          results.data());
      search->ApplyScorerWave(s, results.data());
      extensions.clear();
    }
  }
}

}  // namespace sherpa_onnx

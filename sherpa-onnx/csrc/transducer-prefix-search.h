// sherpa-onnx/csrc/transducer-prefix-search.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_TRANSDUCER_PREFIX_SEARCH_H_
#define SHERPA_ONNX_CSRC_TRANSDUCER_PREFIX_SEARCH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/scored-prefix-trie.h"
#include "sherpa-onnx/csrc/scorer.h"

// Prefix beam search for one utterance of a transducer model, with scorer
// fusion. The recognizer's probabilities depend on the hypothesis, so the
// caller runs the model for the hypotheses the search asks about.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// How a transition moves through the encoder's frames.
enum class TransducerTopology {
  // Blank or one token, then the next frame: the "modified" beam search
  // used for stateless-decoder transducers.
  kOneSymbolPerFrame,
  // A token stays in the frame, so several can be emitted there; blank
  // moves to the next frame.
  kRnnt,
  // Token-and-duration: each transition also chooses how many frames to
  // advance, possibly none after a token.
  kTdt,
};

struct TransducerPrefixSearchOptions {
  TransducerTopology topology = TransducerTopology::kOneSymbolPerFrame;

  // Hypothesis beam: at most this many output prefixes survive each pruning.
  int32_t max_active = 4;
  // A prefix survives only within this score of the best.
  float beam = std::numeric_limits<float>::infinity();

  // Per hypothesis, only its most probable non-blank tokens extend it.
  int32_t max_token_candidates = 8;

  // kRnnt and kTdt: at most this many expansions within one frame; the last
  // allows only transitions that leave the frame.
  int32_t max_symbols_per_frame = 10;
  // kTdt: the largest number of frames a transition can advance, so the
  // model gives max_duration + 1 duration scores. At most kMaxDuration.
  int32_t max_duration = 4;

  // Added to a prefix's score once per unit of its length.
  float length_bonus = 0;
  LengthBonusUnit length_bonus_unit = LengthBonusUnit::kToken;

  // The character a character scorer is given between words.
  UnitId word_boundary_char = ' ';

  // kOneSymbolPerFrame only, to reproduce modified_beam_search: each frame
  // keeps the max_active best single steps (a hypothesis and blank or one
  // token) before equal outputs are combined, where the search otherwise
  // combines every step into its output first and keeps the max_active best
  // outputs. A step's rank is its hypothesis's score plus the step's log
  // probability; the token's own scorer scores count from the next frame.
  bool prune_steps = false;
  // With prune_steps: a token that is taken as blank, as that decoder takes
  // the unknown-word token. Negative for none.
  int32_t blank_like_token = -1;

  // If not negative, Best and Finish rank hypotheses by score divided by
  // (this + number of tokens), as modified_beam_search ranks its final
  // hypotheses with its leading context counted. Pruning during the search
  // and the scores reported are unaffected.
  int32_t rank_length_offset = -1;

  // kTdt and kRnnt waiting lists: an arrival for a later frame waits in
  // that frame's list, which without this is pruned only when the frame
  // begins. With prune_waiting, every round's end also cuts each waiting
  // list to its best waiting_max_active within waiting_beam of its best
  // (0 for either: twice max_active, half of beam, where a survival trace
  // of Parakeet TDT on LibriSpeech kept every result's dominant path). The
  // next frame's list is never cut, since that frame's pruning at
  // max_active within beam follows; so only TDT, whose durations wait
  // further ahead, is affected. A cut entry that later receives probability
  // again rejoins. Not applied with prune_steps, which must reproduce
  // modified_beam_search.
  bool prune_waiting = true;
  int32_t waiting_max_active = 0;
  float waiting_beam = 0;

  // Survival trace, off by default; adds bytes to every prefix's record and
  // a sort of every waiting list per round only when on. For each result it
  // records what each pruning limit had to allow to keep it
  // (TransducerPrefixHypothesis::required), and at the end of every round
  // it ranks each list of arrivals waiting for a later frame, for a policy
  // that would cut those lists then (TransducerWaitHistogram).
  bool trace_survival = false;
};

// What it took to keep a hypothesis's probability: over every queue it
// passed through, the largest gap below that queue's best score and the
// lowest rank (from 1) at which it was kept. "gap" and "rank" are for the
// search's own pruning when a queue is used; "wait_gap" and "wait_rank" for
// the waiting lists of later frames, ranked at the end of each round.
struct TransducerSurvivalNeed {
  float gap = 0;
  int32_t rank = 0;
  float wait_gap = 0;
  int32_t wait_rank = 0;

  // The most demanding of the two, field by field.
  void Absorb(const TransducerSurvivalNeed &other);
};

// With trace_survival: at the end of every round, each entry of each list
// waiting for a later frame, counted by its rank and gap in its list. Entry
// (r, g) counts entries whose rank is at most kRanks[r] and above
// kRanks[r - 1], and whose gap is at most kGaps[g] and above kGaps[g - 1];
// the last of each is unbounded. Summing the counts with r <= R and g <= G
// over "rounds" gives the mean number of waiting entries a policy that cut
// each list at the end of every round to its best kRanks[R] within kGaps[G]
// of its best would have kept.
struct TransducerWaitHistogram {
  static constexpr int32_t kNumRanks = 12;
  static constexpr int32_t kNumGaps = 11;
  static constexpr int32_t kRanks[kNumRanks] = {
      1,  2,   4,   8,   16,   32,
      64, 128, 256, 512, 1024, std::numeric_limits<int32_t>::max()};
  static constexpr float kGaps[kNumGaps] = {
      1, 2, 3, 4, 6, 8, 10, 12, 16, 24, std::numeric_limits<float>::infinity()};

  std::vector<int64_t> counts =
      std::vector<int64_t>(static_cast<size_t>(kNumRanks) * kNumGaps, 0);
  int64_t rounds = 0;
  int64_t peak_waiting = 0;  // The most waiting entries at a round's end.

  // How prefixes of different lengths meet at one frame: an entry's length
  // in tokens minus that of its list's best, offset by kMaxOffset and
  // clamped to [0, 2 * kMaxOffset]. "waiting_lengths" counts the waiting
  // lists' entries at the end of every round (as "counts" does), and
  // "kept_lengths" and "dropped_lengths" the arrivals a frame's pruning
  // kept and dropped, against the best arrival.
  static constexpr int32_t kMaxOffset = 8;
  std::vector<int64_t> waiting_lengths =
      std::vector<int64_t>(2 * kMaxOffset + 1);
  std::vector<int64_t> kept_lengths = std::vector<int64_t>(2 * kMaxOffset + 1);
  std::vector<int64_t> dropped_lengths =
      std::vector<int64_t>(2 * kMaxOffset + 1);

  void Add(int32_t rank, float gap);
  static void AddLength(int32_t offset, std::vector<int64_t> *lengths);
  void Merge(const TransducerWaitHistogram &other);
};

struct TransducerPrefixHypothesis {
  std::vector<int32_t> tokens;
  // For each token, the frame whose emission of it carried the most
  // probability.
  std::vector<int32_t> timestamps;
  // Ranking score: model + weighted scorer scores + length bonus.
  float score = 0;
  // Summed log probability of the output's retained alignments.
  float model_log_prob = 0;
  // Raw, unweighted log score of each scorer, in scorer order.
  std::vector<float> scorer_log_scores;

  // With TransducerPrefixSearchOptions::trace_survival (zero without it).
  //
  // "required": what every path that added probability to this result
  // needed. A search whose max_active is at least required.rank and whose
  // beam is at least required.gap gives this result with this score; one
  // that also cut each waiting list at the end of every round to its best
  // required.wait_rank within required.wait_gap of its best would too, to
  // first order (cutting other entries changes what else reaches later
  // queues). Every path counts however little it added, so this is more
  // than a search needs.
  //
  // "required_dominant": the same along the result's dominant path, the one
  // that at every queue followed the contribution that brought the most
  // probability. Keeping it keeps the result, minus at most the probability
  // of the paths it does not cover.
  TransducerSurvivalNeed required;
  TransducerSurvivalNeed required_dominant;
};

// One utterance's search. Per encoder frame the caller repeats rounds:
//   for each hypothesis in Frontier(), run the model on that frame;
//   Expand() with the resulting log probabilities;
//   for each scorer, while GatherScorerWave appends extensions: run
//   Scorer::Advance on them and ApplyScorerWave;
//   EndRound(), and stop when it returns false;
// then EndFrame(). Prefixes that reach the same token sequence at the same
// frame are one hypothesis whose probabilities add.
class TransducerPrefixSearch {
 public:
  static constexpr int32_t kMaxDuration = 7;

  // "scorers" and "vocabulary" must outlive the search. One weight per scorer.
  TransducerPrefixSearch(const TransducerPrefixSearchOptions &options,
                         std::shared_ptr<const SearchVocabulary> vocabulary,
                         std::vector<Scorer *> scorers,
                         std::vector<float> scorer_weights);

  TransducerPrefixSearch(const TransducerPrefixSearch &) = delete;
  TransducerPrefixSearch &operator=(const TransducerPrefixSearch &) = delete;

  // The hypotheses to run the model for now, all at frame Frame().
  const std::vector<uint32_t> &Frontier() const { return frontier_; }
  int32_t Frame() const { return frame_; }

  // What the model needs to know about a frontier hypothesis.
  // Its last "size" tokens, oldest first, padded in front with "pad".
  void ContextOf(uint32_t hypothesis, int32_t size, int64_t pad,
                 int64_t *context) const;
  int32_t LastTokenOf(uint32_t hypothesis) const;  // -1 for the empty prefix.
  int32_t LengthOf(uint32_t hypothesis) const;     // In tokens.
  uint32_t ParentOf(uint32_t hypothesis) const;
  // Changes when a hypothesis index is reused for another prefix, so a
  // caller can keep per-hypothesis model state in arrays by index.
  uint32_t SerialOf(uint32_t hypothesis) const;

  // "log_probs" is [Frontier().size(), vocabulary size]: each hypothesis's
  // log probability of blank and of every token at this frame. For kTdt,
  // "duration_log_probs" is [Frontier().size(), max_duration + 1]; null
  // otherwise.
  void Expand(const float *log_probs, const float *duration_log_probs);
  bool GatherScorerWave(int32_t scorer,
                        std::vector<ScorerExtension> *extensions);
  void ApplyScorerWave(int32_t scorer, const ScorerAdvanceResult *results);
  // Returns true if Frontier() now holds hypotheses to expand in this frame.
  bool EndRound();
  // Moves to the next frame; Frontier() is then the hypotheses arriving there.
  void EndFrame();

  // The best prefix so far, without end-of-input scores.
  TransducerPrefixHypothesis Best() const;
  // Scores the end of input and returns up to nbest hypotheses, best first.
  // Call after EndFrame of the last frame.
  std::vector<TransducerPrefixHypothesis> Finish(int32_t nbest = 1);
  // Finish for several searches created with the same scorers, with one
  // end-of-input call per scorer serving all of them.
  static std::vector<std::vector<TransducerPrefixHypothesis>> FinishTogether(
      const std::vector<TransducerPrefixSearch *> &searches, int32_t nbest = 1);

  int32_t NumLiveRecords() const { return trie_.NumLiveNodes(); }
  // With trace_survival; empty otherwise.
  const TransducerWaitHistogram &WaitHistogram() const { return waiting_; }

 private:
  using Node = ScoredPrefixTrie::Node;
  struct State;
  struct QueueNeed;
  struct Trace;
  static const size_t kTraceOffset;  // Of the Trace in a prefix's record.

  State *StateOf(Node *node) const;
  Trace *TraceOf(Node *node) const;  // With trace_survival only.
  void Pin(uint32_t index);
  void Unpin(uint32_t index);
  // "need" is the source queue's, with trace_survival; null otherwise.
  void AddAgain(uint32_t index, float log_mass, const QueueNeed *need);
  void AddArrival(uint32_t index, int32_t frames_ahead, float log_mass,
                  const QueueNeed *need);
  // With trace_survival, at the end of a round: ranks each waiting list.
  void RankWaiting();
  // What a hypothesis's probability needed over the queues holding it: on
  // every path, and on its dominant path.
  void NeedOf(Node *node, TransducerSurvivalNeed *all,
              TransducerSurvivalNeed *dominant) const;
  float LengthBonus(const Node &node, bool at_end) const;
  // What results are ordered by: "score", or that over length.
  float RankKey(const Node &node, float score) const;
  // Keeps the best of "candidates" by the probability each holds in "slot"
  // (an arrival slot, or kAgainSlot) plus scorer scores and bonus, and
  // clears and unpins the others. Returns the kept ones, best first.
  // "to_frontier": they become the frontier, under max_active and beam;
  // otherwise they keep waiting, under the waiting list's limits.
  static constexpr int32_t kAgainSlot = -1;
  std::vector<uint32_t> Prune(const std::vector<uint32_t> &candidates,
                              int32_t slot, bool to_frontier = true);
  // Every hypothesis holding probability, with its total log mass.
  void Outstanding(std::vector<uint32_t> *nodes,
                   std::vector<float> *log_mass) const;
  // The hypotheses that can be results, with their log mass.
  void Finishable(std::vector<uint32_t> *nodes,
                  std::vector<float> *log_mass) const;
  std::vector<TransducerPrefixHypothesis> RankFinished(
      const std::vector<uint32_t> &scored,
      const std::vector<float> &scored_mass,
      const std::vector<float> &final_scores, const std::vector<bool> &accepted,
      int32_t nbest) const;
  TransducerPrefixHypothesis MakeHypothesis(uint32_t index, float score,
                                            float log_mass,
                                            const float *final_scores) const;

  TransducerPrefixSearchOptions options_;
  int32_t ring_size_ = 2;  // Frames ahead that can hold probability, plus one.
  ScoredPrefixTrie trie_;

  int32_t frame_ = 0;
  int32_t round_ = 0;
  std::vector<uint32_t> frontier_;
  std::vector<uint32_t> again_;  // Received probability for the next round.
  std::vector<std::vector<uint32_t>> arriving_;  // By frame modulo ring size.
  std::vector<int32_t> candidates_;
  // With prune_steps: the chosen steps, as (rank, frontier position, token).
  struct Step {
    float key;
    uint32_t position;
    int32_t token;
  };
  std::vector<Step> steps_;
  TransducerWaitHistogram waiting_;
};

// After Expand: runs every scorer's waves for one search.
void RunScorerWaves(TransducerPrefixSearch *search,
                    const std::vector<Scorer *> &scorers);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TRANSDUCER_PREFIX_SEARCH_H_

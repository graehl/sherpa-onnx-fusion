// sherpa-onnx/csrc/ctc-prefix-search.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_CTC_PREFIX_SEARCH_H_
#define SHERPA_ONNX_CSRC_CTC_PREFIX_SEARCH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/scored-prefix-trie.h"
#include "sherpa-onnx/csrc/scorer.h"

// CTC prefix beam search for one utterance, with scorer fusion.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

using CtcSearchToken = SearchToken;
using CtcSearchVocabulary = SearchVocabulary;

enum class CtcAlignmentMode {
  kSum,  // An output's score is the summed probability of its alignments.
  // An output's score is the probability of its best alignment. When every
  // scorer's ScorerInfo::recombines is set, a hypothesis is dropped as soon
  // as another with the same last token and equal scorer states scores at
  // least as well. That cannot change the best result, but results after
  // the first are then no longer each output's own best alignment.
  kMax,
};

struct CtcPrefixSearchOptions {
  CtcAlignmentMode alignment = CtcAlignmentMode::kSum;

  // In max mode, drop a hypothesis that another with the same future
  // matches or beats (see CtcAlignmentMode::kMax). Off only for comparison.
  bool recombine = true;

  // Experimental, sum mode: drop a hypothesis that another with the same
  // future matches or beats in both of its sums, as max mode does. Unlike
  // there it is an approximation: the dropped prefix's probability so far
  // is lost if later frames bring it back.
  bool prune_dominated_in_sum = false;

  // Hypothesis beam: at most this many output prefixes survive a frame.
  int32_t max_active = 8;
  // A prefix survives only within this score of the frame's best.
  float beam = std::numeric_limits<float>::infinity();

  // Per frame, only the most probable non-blank tokens extend a prefix.
  int32_t max_token_candidates = 8;

  // Added to a prefix's score once per unit of its length.
  float length_bonus = 0;
  LengthBonusUnit length_bonus_unit = LengthBonusUnit::kToken;

  // The character a character scorer is given between words.
  UnitId word_boundary_char = ' ';

  // Optional traces, off by default; each adds bytes to every hypothesis's
  // record and work per frame only when on.
  //
  // Survival: for each result, how wide the score-gap beam and the
  // hypothesis beam had to be to keep its best alignment at every frame.
  bool trace_survival = false;
};

struct CtcPrefixHypothesis {
  std::vector<int32_t> tokens;
  // For each token, the frame whose emission of it carried the most
  // probability.
  std::vector<int32_t> timestamps;
  // Ranking score: acoustic + weighted scorer scores + length bonus.
  float score = 0;
  // Log probability of the output's retained alignments (sum or best).
  float acoustic_log_prob = 0;
  // Raw, unweighted log score of each scorer, in scorer order.
  std::vector<float> scorer_log_scores;

  // With CtcPrefixSearchOptions::trace_survival: over the frames of the
  // prefixes this output's best alignment passed through, the largest gap
  // below the frame's best score and the lowest rank at which it was kept.
  // A search whose score-gap beam exceeds the first, or whose hypothesis
  // beam is at least the second, gives the same result and score. In max
  // mode these are what the output's one best alignment needed. In sum mode
  // they are what every alignment that added to the output needed, however
  // little it added, so they are larger than a search needs in practice.
  // Zero without the trace.
  float required_score_beam = 0;
  int32_t required_max_active = 0;
};

// What it took to keep an alignment alive: over the frames of the prefixes it
// passed through, the largest gap below the frame's best score and the lowest
// rank (counting from 1) at which it was kept.
struct SurvivalNeed {
  float gap;
  int32_t rank;
};

// One utterance's search. Frames arrive in any chunking; chunk boundaries do
// not change the result.
//
// A frame is processed in phases so that a caller decoding several
// utterances can put their scorer work in one batch:
//   BeginFrame, then for each scorer, while GatherScorerWave appends
//   extensions: run Scorer::Advance on them and ApplyScorerWave; then EndFrame.
// Decode does exactly this for one utterance.
class CtcPrefixSearch {
 public:
  // "scorers" and "vocabulary" must outlive the search. One weight per scorer.
  CtcPrefixSearch(const CtcPrefixSearchOptions &options,
                  std::shared_ptr<const CtcSearchVocabulary> vocabulary,
                  std::vector<Scorer *> scorers,
                  std::vector<float> scorer_weights);
  ~CtcPrefixSearch();

  CtcPrefixSearch(const CtcPrefixSearch &) = delete;
  CtcPrefixSearch &operator=(const CtcPrefixSearch &) = delete;

  // Processes num_frames rows of vocabulary-size log probabilities.
  void Decode(const float *log_probs, int32_t num_frames);

  void BeginFrame(const float *log_probs);
  // Appends this utterance's next extensions for scorer "scorer" and returns
  // true, or returns false when that scorer has no work left in this frame.
  bool GatherScorerWave(int32_t scorer,
                        std::vector<ScorerExtension> *extensions);
  // "results" are for the extensions the last GatherScorerWave appended.
  void ApplyScorerWave(int32_t scorer, const ScorerAdvanceResult *results);
  void EndFrame();

  // The best prefix so far, without end-of-input scores. Empty tokens before
  // any output.
  CtcPrefixHypothesis Best() const;

  // Scores the end of input and returns up to nbest hypotheses, best first.
  // Empty if every scorer-accepted hypothesis was pruned. The search stays
  // usable for further frames.
  std::vector<CtcPrefixHypothesis> Finish(int32_t nbest = 1);
  // Finish for several searches created with the same scorers, with one
  // end-of-input call per scorer serving all of them.
  static std::vector<std::vector<CtcPrefixHypothesis>> FinishTogether(
      const std::vector<CtcPrefixSearch *> &searches, int32_t nbest = 1);

  int32_t NumFramesDecoded() const { return frame_; }
  // Hypotheses dropped so far because another with the same future scored
  // at least as well (max alignment, when every scorer recombines).
  int64_t NumRecombined() const { return num_recombined_; }
  // Work counters since construction: hypotheses kept after pruning,
  // summed over frames, and new prefixes sent to the scorers.
  int64_t NumHypothesesKept() const { return num_kept_; }
  int64_t NumPrefixesScored() const { return trie_.NumNodesScored(); }
  // Prefix records currently held, for tests of reclamation.
  int32_t NumLiveRecords() const { return trie_.NumLiveNodes(); }

 private:
  using Node = ScoredPrefixTrie::Node;
  struct CtcState;
  struct SurvivalTrace;

  CtcState *Ctc(Node *node) const;
  SurvivalTrace *SurvivalOf(Node *node) const;

  void Touch(uint32_t index);
  size_t KeepUndominated(std::vector<uint32_t> *order, size_t limit);
  void NoteContribution(float term, const SurvivalNeed &from, float *best_term,
                        SurvivalNeed *next) const;
  float Acoustic(float blank, float nonblank) const;
  float Accumulate(float sum, float term) const;
  float LengthBonus(const Node &node, bool at_end) const;
  std::vector<CtcPrefixHypothesis> RankFinished(
      const std::vector<float> &final_scores, const std::vector<bool> &accepted,
      int32_t nbest) const;
  CtcPrefixHypothesis MakeHypothesis(uint32_t index, float score,
                                     float acoustic,
                                     const float *final_scores) const;

  CtcPrefixSearchOptions options_;
  // Each prefix's record holds a CtcState and, with trace_survival, a
  // SurvivalTrace after it.
  size_t survival_offset_ = 0;
  ScoredPrefixTrie trie_;

  std::vector<uint32_t> active_;
  int32_t frame_ = 0;
  bool recombine_ = false;
  int64_t num_recombined_ = 0;
  int64_t num_kept_ = 0;

  // Per-frame work.
  std::vector<uint32_t> touched_;
  std::vector<float> touched_scores_;
  std::vector<int32_t> candidates_;
  // KeepUndominated's scratch.
  std::vector<uint32_t> kept_, dropped_, future_heads_, future_chain_;
};

// Between BeginFrame and EndFrame of each search in "searches": runs every
// scorer's waves, each scorer call serving all the searches. Every search
// must have been created with "scorers", in that order.
void RunScorerWavesTogether(const std::vector<CtcPrefixSearch *> &searches,
                            const std::vector<Scorer *> &scorers);

// Decodes frames of several utterances together, so each scorer call
// serves every utterance that has a frame at that position. log_probs is
// [searches.size(), num_frames, vocab_size]; lengths[i] is utterance i's
// number of valid frames, or lengths is null when all have num_frames.
// Every search must have been created with "scorers", in that order.
void DecodeFramesTogether(const std::vector<CtcPrefixSearch *> &searches,
                          const std::vector<Scorer *> &scorers,
                          const float *log_probs, int32_t num_frames,
                          int32_t vocab_size, const int64_t *lengths);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CTC_PREFIX_SEARCH_H_

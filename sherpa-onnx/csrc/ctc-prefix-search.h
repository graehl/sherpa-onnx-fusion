// sherpa-onnx/csrc/ctc-prefix-search.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_CTC_PREFIX_SEARCH_H_
#define SHERPA_ONNX_CSRC_CTC_PREFIX_SEARCH_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx/csrc/scorer.h"

// CTC prefix beam search for one utterance, with scorer fusion.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// What the search needs to know about one recognizer token.
struct CtcSearchToken {
  // False for a token the search never emits, such as an unknown-word token.
  bool emit = true;
  // The token's text begins a new word (its piece begins with a space).
  bool starts_word = false;
  // Unicode scalars of the token's text, without that leading space.
  std::vector<UnitId> chars;
};

struct CtcSearchVocabulary {
  int32_t blank_id = 0;
  std::vector<CtcSearchToken> tokens;  // Indexed by token ID.
};

enum class CtcAlignmentMode {
  kSum,  // An output's score is the summed probability of its alignments.
  kMax,  // An output's score is the probability of its best alignment.
};

enum class LengthBonusUnit { kToken, kWord };

struct CtcPrefixSearchOptions {
  CtcAlignmentMode alignment = CtcAlignmentMode::kSum;

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

  int32_t NumFramesDecoded() const { return frame_; }
  // Prefix records currently held, for tests of reclamation.
  int32_t NumLiveRecords() const { return num_live_records_; }

 private:
  struct Record;
  struct Pending;

  Record *Rec(uint32_t index) const;
  std::byte *StateOf(Record *record, int32_t scorer) const;
  float *SubtotalsOf(Record *record) const;

  uint32_t NewRecord(uint32_t parent, int32_t token);
  void FreeRecord(uint32_t index);
  void ReleaseIfUnused(uint32_t index);
  uint32_t FindOrCreateChild(uint32_t parent, int32_t token);
  void Touch(uint32_t index);
  void QueueScoring(uint32_t index);
  void FinishScoring();
  float Acoustic(float blank, float nonblank) const;
  float Accumulate(float sum, float term) const;
  float LengthBonus(const Record &record, bool at_end) const;
  CtcPrefixHypothesis MakeHypothesis(uint32_t index, float score,
                                     float acoustic,
                                     const float *final_scores) const;

  CtcPrefixSearchOptions options_;
  std::shared_ptr<const CtcSearchVocabulary> vocabulary_;
  std::vector<Scorer *> scorers_;
  std::vector<float> weights_;
  std::vector<std::unique_ptr<ScorerUtterance>> utterances_;

  // Record layout: header, one float subtotal per scorer, then each
  // scorer's state bytes, all 8-byte aligned.
  size_t subtotals_offset_ = 0;
  std::vector<size_t> state_offsets_;
  size_t record_bytes_ = 0;

  // Records live in fixed-size blocks, so a record's address is stable and
  // records are addressed by index.
  static constexpr uint32_t kBlockShift = 8;
  std::vector<std::unique_ptr<uint64_t[]>> blocks_;
  uint32_t num_records_ = 0;
  std::vector<uint32_t> free_records_;
  int32_t num_live_records_ = 0;
  std::unordered_map<uint64_t, uint32_t> children_;  // (parent, token) -> index

  std::vector<uint32_t> active_;
  int32_t frame_ = 0;

  // Per-frame work.
  std::vector<uint32_t> touched_;
  std::vector<float> touched_scores_;
  std::vector<int32_t> candidates_;
  std::vector<uint32_t> new_records_;
  std::vector<std::vector<Pending>> pending_;   // Per scorer.
  std::vector<std::vector<uint32_t>> wave_;     // Pending indices sent.
  std::vector<std::vector<uint64_t>> scratch_;  // Intermediate states.
  std::vector<UnitId> units_;
};

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

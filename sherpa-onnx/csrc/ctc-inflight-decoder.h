// sherpa-onnx/csrc/ctc-inflight-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_CTC_INFLIGHT_DECODER_H_
#define SHERPA_ONNX_CSRC_CTC_INFLIGHT_DECODER_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "sherpa-onnx/csrc/ctc-prefix-search.h"
#include "sherpa-onnx/csrc/scorer.h"

// CTC prefix beam search over many utterances at once, with inflight
// batching: utterances are admitted, fed, finished, and cancelled
// individually while others stay active, and each scorer call serves all
// utterances that have a frame ready.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// Identifies an utterance for the life of a decoder; never reused.
struct UtteranceHandle {
  uint64_t id = 0;
};

enum class DecodeStatus {
  kActive,          // Accepting or decoding input.
  kComplete,        // Input ended and the final result is ready.
  kNoAcceptedPath,  // Input ended and every hypothesis was rejected.
  kCancelled,
};

struct CtcInflightDecoderOptions {
  CtcPrefixSearchOptions search;
  // Default weight of each scorer, in scorer order.
  std::vector<float> scorer_weights;

  // Admission capacity.
  int32_t max_active_utterances = 64;
  // Bound on log probabilities accepted and not yet decoded, in bytes.
  size_t max_queued_input_bytes = size_t{64} << 20;

  // Frames each ready utterance advances per Step. Bounds scheduling work
  // per call; it is not a time bound.
  int32_t steps_per_utterance = 1;
};

struct UtteranceOptions {
  // Empty means the decoder's defaults; otherwise one weight per scorer.
  std::vector<float> scorer_weights;
  int32_t nbest = 1;
};

struct StepProgress {
  int32_t frames_decoded = 0;  // Summed over utterances, in this Step.
  // Some active utterance still has queued frames.
  bool has_ready_work = false;
  // Some active utterance has no queued frames and its input is not ended.
  bool waiting_for_input = false;
};

// One coordinator calls these methods serially. Invalid or released handles
// throw std::out_of_range; a wrong number of scorer weights, and Release of
// an utterance that is still active, throw std::invalid_argument.
class CtcInflightDecoder {
 public:
  // "vocabulary" gives the number of tokens per frame of log probabilities.
  CtcInflightDecoder(const CtcInflightDecoderOptions &options,
                     std::shared_ptr<const CtcSearchVocabulary> vocabulary,
                     std::vector<std::shared_ptr<Scorer>> scorers);
  ~CtcInflightDecoder();

  // Admits an utterance, or returns nullopt when at capacity.
  std::optional<UtteranceHandle> TryAdd(const UtteranceOptions &options = {});

  // Queues num_frames rows of log probabilities, copying them. Returns false
  // and takes nothing when that would exceed max_queued_input_bytes or the
  // utterance is not active or its input has ended. Throws
  // std::invalid_argument for a chunk that could never fit.
  bool TryAccept(UtteranceHandle handle, const float *log_probs,
                 int32_t num_frames);

  // No further input: the utterance completes once its queue is decoded.
  void InputFinished(UtteranceHandle handle);

  // Stops work on the utterance and frees its search at once.
  void Cancel(UtteranceHandle handle);

  // Decodes up to steps_per_utterance frames of every utterance that has
  // frames queued, batching scorer calls across them, and completes
  // utterances whose input has ended.
  StepProgress Step();

  DecodeStatus Status(UtteranceHandle handle) const;

  // While active: the best prefix so far, without end-of-input scores.
  // Once complete: up to nbest final hypotheses, best first. Otherwise empty.
  std::vector<CtcPrefixHypothesis> GetResult(UtteranceHandle handle) const;

  // Forgets a complete, rejected, or cancelled utterance.
  void Release(UtteranceHandle handle);

  int32_t NumActiveUtterances() const { return num_active_; }

 private:
  struct Utterance;

  Utterance &Get(UtteranceHandle handle) const;
  void DropWork(Utterance *utterance);

  CtcInflightDecoderOptions options_;
  std::shared_ptr<const CtcSearchVocabulary> vocabulary_;
  std::vector<std::shared_ptr<Scorer>> scorers_;
  std::vector<Scorer *> scorer_pointers_;

  // Only unreleased handles are retained; IDs are never reused.
  std::map<uint64_t, std::unique_ptr<Utterance>> utterances_;
  uint64_t next_handle_ = 0;
  int32_t num_active_ = 0;
  size_t queued_bytes_ = 0;

  // Reused by Step.
  std::vector<Utterance *> ready_;
  std::vector<CtcPrefixSearch *> ready_searches_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CTC_INFLIGHT_DECODER_H_

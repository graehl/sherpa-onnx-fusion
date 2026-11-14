// sherpa-onnx/csrc/nemo-transducer-prefix-segment.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_NEMO_TRANSDUCER_PREFIX_SEGMENT_H_
#define SHERPA_ONNX_CSRC_NEMO_TRANSDUCER_PREFIX_SEGMENT_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/transducer-prefix-search.h"

namespace sherpa_onnx {

// What a prefix search needs from a NeMo transducer (RNN-T or TDT), so the
// offline and streaming decoders share the search below.
struct NeMoTransducerCalls {
  // The prediction network for several hypotheses: tokens is int32 [N, 1]
  // and each state tensor [layers, N, hidden]. Returns its output, [N, ...],
  // and the new states.
  std::function<std::pair<Ort::Value, std::vector<Ort::Value>>(
      Ort::Value tokens, std::vector<Ort::Value> states)>
      run_decoder;
  // encoder_out is [N, encoder_dim, 1]; returns logits, N rows.
  std::function<Ort::Value(Ort::Value encoder_out, Ort::Value decoder_out)>
      run_joiner;
  // The prediction network's initial states for one hypothesis.
  std::function<std::vector<Ort::Value>()> init_states;
  OrtAllocator *allocator = nullptr;
  int32_t vocab_size = 0;  // Blank is the last token.
  bool tdt = false;
  // Cleared the first time the model refuses several hypotheses in one
  // call; they then go one at a time.
  bool joiner_batches = true;
  bool decoder_batches = true;
};

// One utterance's, or one streaming segment's, prefix search on a NeMo
// transducer, with the prediction network's output and state for each
// hypothesis it has asked about. The prediction network and the joiner each
// run once per round for the hypotheses that need them.
class NeMoTransducerPrefixSegment {
 public:
  // "calls" and "fusion" must outlive the segment.
  NeMoTransducerPrefixSegment(NeMoTransducerCalls *calls,
                              const TransducerFusion *fusion,
                              const FusionConfig &config, int32_t max_active,
                              float blank_penalty);
  ~NeMoTransducerPrefixSegment();

  // Searches num_frames more frames, [num_frames, encoder_dim].
  void Decode(const float *encoder_out, int32_t num_frames,
              int32_t encoder_dim);

  // Null before the first Decode.
  TransducerPrefixSearch *Search() const { return search_.get(); }

 private:
  struct Prediction;

  Prediction &PredictionOf(uint32_t hypothesis);
  void PredictTogether(const std::vector<uint32_t> &frontier);
  void Join(const float *frame, int32_t encoder_dim,
            const std::vector<uint32_t> &frontier, int32_t first, int32_t count,
            std::vector<float> *log_probs, std::vector<float> *durations);

  NeMoTransducerCalls *calls_;
  const TransducerFusion *fusion_;
  FusionConfig config_;
  int32_t max_active_;
  float blank_penalty_;
  int32_t num_durations_ = 0;
  std::unique_ptr<TransducerPrefixSearch> search_;
  std::vector<Prediction> predictions_;  // By hypothesis index.
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_NEMO_TRANSDUCER_PREFIX_SEGMENT_H_

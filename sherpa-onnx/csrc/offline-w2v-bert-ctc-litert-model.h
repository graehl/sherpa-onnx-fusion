// sherpa-onnx/csrc/offline-w2v-bert-ctc-litert-model.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_LITERT_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_LITERT_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// The w2v-BERT 2.0 CTC model of offline-w2v-bert-ctc-model.h run through
// LiteRT from a .tflite file instead of through ONNX Runtime, so that a
// phone's accelerator can run it. Built with -DSHERPA_ONNX_ENABLE_LITERT=ON
// and chosen when --w2v-bert-ctc names a file ending in .tflite, or several,
// comma separated.
//
// A LiteRT graph for an accelerator has fixed shapes, and costs the same
// whatever the utterance's length. Each signature of each file given is one
// length P: it takes, by name, "input_features" float32 (1, P, 160) and
// "input_lengths" int32 (1), and returns "logits" float32 (1, F, vocab_size)
// and "logits_lengths" int32 (1), masking the padding itself. F is P or
// ceil(P / 2), with the same output stride across all signatures. A file of
// several signatures holds the weights once. An utterance runs in the
// shortest length that holds it; a longer one runs in overlapping windows
// of one length (--litert-window W:O:S in seconds: windows of the shortest
// graph of at least W seconds, default the longest, evenly spaced with at
// least O seconds of overlap, each interior one moved by up to S seconds to
// put its edges where the filterbank energy is lowest; without
// --litert-window, --encoder.window.seconds, overlap and slack). By default
// the windows' logits are blended per frame with weight 1 + distance from the
// window's nearer edge, and each run's features are normalized over that
// run's own frames, so NormalizeFeatures leaves them alone;
// --encoder.window.normalize, weight and blend choose otherwise
// (topics/chunked-encoding.md).
//
// --provider=npu runs a graph compiled ahead of time for the phone's chip
// (for a Pixel's Tensor processor, with libLiteRtDispatch_GoogleTensor.so
// in --litert-dispatch-dir). Any other provider runs a portable graph on
// LiteRT's CPU kernels.
//
// The model builds its graphs in a LiteRT environment it creates, or, when
// config.litert_environment is set, in that caller-owned one, which it
// never destroys.
class OfflineW2vBertCtcLiteRtModel : public OfflineCtcModel {
 public:
  explicit OfflineW2vBertCtcLiteRtModel(const OfflineModelConfig &config);
  ~OfflineW2vBertCtcLiteRtModel() override;

  // features is (1, T, 80), unnormalized, with features_length int64 (1,).
  // Returns log probabilities (1, F, vocab_size) and the count of valid
  // frames among the F, int64 (1,).
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  int32_t VocabSize() const override;
  OrtAllocator *Allocator() const override;
  int32_t SubsamplingFactor() const override;
  bool SupportBatchProcessing() const override { return false; }

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_LITERT_MODEL_H_

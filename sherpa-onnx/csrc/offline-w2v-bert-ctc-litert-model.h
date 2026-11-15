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
// and chosen when --w2v-bert-ctc names a file ending in .tflite.
//
// A LiteRT graph has fixed shapes. This one takes, by signature name,
// "input_features" float32 (1, P, 160) and "input_lengths" int32 (1), and
// returns "logits" float32 (1, F, vocab_size) and "logits_lengths" int32
// (1). It masks padding itself: an utterance is zero padded to P frames and
// its own frame count passed. One longer than P frames is cut to P, with an
// error logged.
//
// --provider=npu runs a graph compiled ahead of time for the phone's chip
// (for a Pixel's Tensor processor, with libLiteRtDispatch_GoogleTensor.so
// in --litert-dispatch-dir). Any other provider runs a portable graph on
// LiteRT's CPU kernels.
class OfflineW2vBertCtcLiteRtModel : public OfflineCtcModel {
 public:
  explicit OfflineW2vBertCtcLiteRtModel(const OfflineModelConfig &config);
  ~OfflineW2vBertCtcLiteRtModel() override;

  // features is (1, T, 80), normalized, with features_length int64 (1,).
  // Returns log probabilities (1, F, vocab_size) and the count of valid
  // frames among the F, int64 (1,).
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  int32_t VocabSize() const override;
  OrtAllocator *Allocator() const override;
  int32_t SubsamplingFactor() const override { return 2; }
  bool SupportBatchProcessing() const override { return false; }

  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_LITERT_MODEL_H_

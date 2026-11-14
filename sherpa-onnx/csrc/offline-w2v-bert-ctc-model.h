// sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {

// A w2v-BERT 2.0 encoder with a CTC head, exported from Hugging Face
// transformers (Wav2Vec2BertForCTC) with one input, "input_features" of
// shape (N, T, 160), and one output of logits (N, T, vocab_size).
//
// The input is what transformers' SeamlessM4TFeatureExtractor makes: 80
// Kaldi filterbank energies per 10 ms from samples scaled to 16 bits, each
// bin normalized over the utterance to zero mean and unit variance, and
// each two consecutive frames joined into one of 160 values. The recognizer
// supplies the filterbank frames (OfflineModelConfig::w2v_bert_ctc sets its
// options); NormalizeFeatures and Forward do the rest.
//
// The export has no attention mask, so a padded batch would change the
// output; utterances are run one at a time.
class OfflineW2vBertCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineW2vBertCtcModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineW2vBertCtcModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineW2vBertCtcModel() override;

  // features is (1, T, 80), normalized. Returns log probabilities
  // (1, T / 2, vocab_size) and their length, int64 (1,).
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  int32_t VocabSize() const override;
  OrtAllocator *Allocator() const override;
  int32_t SubsamplingFactor() const override { return 2; }
  bool SupportBatchProcessing() const override { return false; }

  // Each bin to zero mean and unit variance over the utterance.
  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_

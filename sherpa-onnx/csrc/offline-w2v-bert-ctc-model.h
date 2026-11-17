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

// Normalizes each of the feat_dim bins of features (num_frames rows) to zero
// mean and unit variance over the utterance, as SeamlessM4TFeatureExtractor
// does.
void NormalizeW2vBertFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim);

// A w2v-BERT 2.0 encoder with a CTC head, exported from Hugging Face
// transformers (Wav2Vec2BertForCTC) with the input "input_features" of
// shape (N, T, 160) and an output of logits (N, T_out, vocab_size).
// The optional subsampling_factor metadata includes feature stacking and
// adapter strides; absent metadata retains the original factor of 2.
//
// The input is what transformers' SeamlessM4TFeatureExtractor makes: 80
// Kaldi filterbank energies per 10 ms from samples scaled to 16 bits, each
// bin normalized over the utterance to zero mean and unit variance, and
// each two consecutive frames joined into one of 160 values. The recognizer
// supplies the filterbank frames (OfflineModelConfig::w2v_bert_ctc sets its
// options); NormalizeFeatures and Forward do the rest.
//
// An export with a second input, "input_lengths" (int64 (N,), each
// utterance's valid frames on the T axis of input_features), masks padding
// inside the graph and is run on padded batches; it may return a second
// output, the valid output frames, int64 (N,). An export with the one input
// has no mask, so a padded batch would change its output, and utterances are
// run one at a time.
class OfflineW2vBertCtcModel : public OfflineCtcModel {
 public:
  explicit OfflineW2vBertCtcModel(const OfflineModelConfig &config);

  template <typename Manager>
  OfflineW2vBertCtcModel(Manager *mgr, const OfflineModelConfig &config);

  ~OfflineW2vBertCtcModel() override;

  // features is (N, T, 80), normalized, with features_length int64 (N,);
  // N is 1 unless SupportBatchProcessing(). Returns log probabilities
  // (N, T_out, vocab_size) and their lengths, int64 (N,).
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) override;

  int32_t VocabSize() const override;
  OrtAllocator *Allocator() const override;
  int32_t SubsamplingFactor() const override;
  bool SupportBatchProcessing() const override;

  // Each bin to zero mean and unit variance over the utterance.
  void NormalizeFeatures(float *features, int32_t num_frames,
                         int32_t feat_dim) const override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_W2V_BERT_CTC_MODEL_H_

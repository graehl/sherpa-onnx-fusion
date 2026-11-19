// sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model.cc
//
// Copyright (c)  2025  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-omnilingual-asr-ctc-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/batching-metadata.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/transpose.h"

namespace sherpa_onnx {

class OfflineOmnilingualAsrCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.omnilingual.model), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.omnilingual.model);
    Init(buf.data(), buf.size());
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    if (takes_lengths_) {
      // The graph masks padding and returns each utterance's valid frames.
      std::array<Ort::Value, 2> inputs = {std::move(features),
                                          std::move(features_length)};
      return sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                        inputs.size(), output_names_ptr_.data(),
                        output_names_ptr_.size());
    }

    auto out_vec =
        sess_->Run({}, input_names_ptr_.data(), &features, 1,
                   output_names_ptr_.data(), output_names_ptr_.size());
    std::vector<int64_t> logits_shape =
        out_vec[0].GetTensorTypeAndShapeInfo().GetShape();

    std::vector<int64_t> num_frames(logits_shape[0], logits_shape[1]);
    if (logits_shape[0] > 1) {
      // A batch the metadata allows without lengths: each utterance's valid
      // frames are its samples through the convolutional frontend.
      const int64_t *samples = features_length.GetTensorData<int64_t>();
      int64_t padded = features.GetTensorTypeAndShapeInfo().GetShape()[1];
      if (FramesOfSamples(padded) != logits_shape[1]) {
        SHERPA_ONNX_LOGE(
            "Omnilingual asr model without input_lengths: %d samples gave "
            "%d frames, not the %d of the wav2vec2 frontend, so a batch "
            "cannot tell each utterance's valid frames",
            static_cast<int32_t>(padded), static_cast<int32_t>(logits_shape[1]),
            static_cast<int32_t>(FramesOfSamples(padded)));
        SHERPA_ONNX_EXIT(-1);
      }
      for (int64_t b = 0; b != logits_shape[0]; ++b) {
        num_frames[b] = FramesOfSamples(samples[b]);
      }
    }

    int64_t shape = logits_shape[0];

    Ort::Value logits_len =
        Ort::Value::CreateTensor<int64_t>(allocator_, &shape, 1);
    std::copy(num_frames.begin(), num_frames.end(),
              logits_len.GetTensorMutableData<int64_t>());

    out_vec.push_back(std::move(logits_len));

    return out_vec;
  }

  int32_t VocabSize() const { return vocab_size_; }

  bool Batching() const { return batching_; }

  OrtAllocator *Allocator() { return allocator_; }

  static void NormalizeFeatures(float *features, int32_t num_frames,
                                int32_t feat_dim) {
    if (num_frames != 1) {
      SHERPA_ONNX_LOGE(
          "Unexpected error in collecting samples for Omnilingual ASR models!");
      return;
    }

    // Map the single-row feature vector
    Eigen::Map<Eigen::ArrayXf> x(features, feat_dim);
    float mean = x.mean();
    // E[x^2] - E[x]^2 cancels catastrophically in float32 when a feature
    // stays near a constant value, so compute the variance from centered
    // values instead.
    float var = (x - mean).square().mean();
    var = std::max(var, 0.0f);
    float inv_stddev = 1.0f / std::sqrt(var + 1e-5f);

    x = (x - mean) * inv_stddev;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    // For models with 1B parameters, weights are saved externally
    // in model.weights
    // We cannot create session from buffer in this case.
    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(env_, model_data,
                                             model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass buffer data or initialize session outside of this "
          "function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // An export with a second input, input_lengths (valid samples of each
    // utterance), masks padding in the graph and has a second output, the
    // valid frames of each utterance. Without metadata, only such an export
    // is batched; one whose metadata says it batches without them gets its
    // valid frames from FramesOfSamples.
    takes_lengths_ = input_names_.size() == 2 && output_names_.size() == 2;

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s\n", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s\n", os.str().c_str());
#endif
    }

    // get vocab size from the output[0].shape, which is (N, T, vocab_size)
    vocab_size_ =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];

    batching_ = BatchingSupported(meta_data, allocator_, takes_lengths_);
  }

  // Output frames of the wav2vec2 convolutional frontend for this many
  // samples (kernels 10,3,3,3,3,2,2, strides 5,2,2,2,2,2,2, no padding),
  // checked against the published export from 400 to 48000 samples.
  static int64_t FramesOfSamples(int64_t n) {
    static constexpr int32_t kKernel[] = {10, 3, 3, 3, 3, 2, 2};
    static constexpr int32_t kStride[] = {5, 2, 2, 2, 2, 2, 2};
    for (int32_t i = 0; i != 7; ++i) n = (n - kKernel[i]) / kStride[i] + 1;
    return n;
  }

 private:
  OfflineModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  int32_t vocab_size_ = 0;
  bool takes_lengths_ = false;
  bool batching_ = false;  // BatchingSupported, defaulting to takes_lengths_
};

OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    Manager *mgr, const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineOmnilingualAsrCtcModel::~OfflineOmnilingualAsrCtcModel() = default;

std::vector<Ort::Value> OfflineOmnilingualAsrCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineOmnilingualAsrCtcModel::VocabSize() const {
  return impl_->VocabSize();
}

bool OfflineOmnilingualAsrCtcModel::SupportBatchProcessing() const {
  return impl_->Batching();
}

OrtAllocator *OfflineOmnilingualAsrCtcModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineOmnilingualAsrCtcModel::NormalizeFeatures(float *features,
                                                      int32_t num_frames,
                                                      int32_t feat_dim) const {
  return impl_->NormalizeFeatures(features, num_frames, feat_dim);
}

#if __ANDROID_API__ >= 9
template OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineOmnilingualAsrCtcModel::OfflineOmnilingualAsrCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/offline-w2v-bert-ctc-model.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h"

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

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

// Two filterbank frames become one input frame.
constexpr int32_t kStack = 2;

}  // namespace

class OfflineW2vBertCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.w2v_bert_ctc), sess_opts_);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.w2v_bert_ctc);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    auto shape = features.GetTensorTypeAndShapeInfo().GetShape();
    auto batch = static_cast<int32_t>(shape[0]);
    auto num_frames = static_cast<int32_t>(shape[1]);
    auto feat_dim = static_cast<int32_t>(shape[2]);
    // An odd last frame is dropped, as the feature extractor drops it.
    int32_t stacked_frames = num_frames / kStack;
    size_t row = static_cast<size_t>(stacked_frames) * feat_dim * kStack;

    std::array<int64_t, 3> stacked_shape = {batch, stacked_frames,
                                            feat_dim * kStack};
    // Each utterance's frames are already consecutive in memory: the same
    // values with a different shape, less the odd frame.
    Ort::Value stacked = Ort::Value::CreateTensor<float>(
        allocator_, stacked_shape.data(), stacked_shape.size());
    const float *p = features.GetTensorData<float>();
    float *q = stacked.GetTensorMutableData<float>();
    for (int32_t b = 0; b != batch; ++b) {
      std::copy(p + static_cast<size_t>(b) * num_frames * feat_dim,
                p + static_cast<size_t>(b) * num_frames * feat_dim + row,
                q + b * row);
    }

    std::array<int64_t, 1> length_shape = {batch};
    Ort::Value length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());
    int64_t *stacked_length = length.GetTensorMutableData<int64_t>();
    const int64_t *frames_length = features_length.GetTensorData<int64_t>();
    for (int32_t b = 0; b != batch; ++b) {
      stacked_length[b] = frames_length[b] / kStack;
    }

    std::vector<Ort::Value> out;
    if (takes_lengths_) {
      // Padding is finite and the model masks it; zero is the value its
      // normalized features have on average.
      for (int32_t b = 0; b != batch; ++b) {
        std::fill(q + b * row + stacked_length[b] * feat_dim * kStack,
                  q + (b + 1) * row, 0.0f);
      }
      std::array<Ort::Value, 2> inputs = {std::move(stacked), View(&length)};
      out =
          sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                     output_names_ptr_.data(), output_names_ptr_.size());
    } else {
      out = sess_->Run({}, input_names_ptr_.data(), &stacked, 1,
                       output_names_ptr_.data(), output_names_ptr_.size());
    }
    auto out_shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
    auto out_frames = static_cast<int32_t>(out_shape[1]);
    auto vocab_size = static_cast<int32_t>(out_shape[2]);
    LogSoftmax(out[0].GetTensorMutableData<float>(), vocab_size,
               batch * out_frames);

    std::vector<Ort::Value> ans;
    ans.push_back(std::move(out[0]));
    if (out.size() > 1) {
      // The model's own count of valid output frames.
      ans.push_back(std::move(out[1]));
    } else {
      // One output frame per stacked input frame.
      for (int32_t b = 0; b != batch; ++b) {
        stacked_length[b] = std::min<int64_t>(stacked_length[b], out_frames);
      }
      ans.push_back(std::move(length));
    }
    return ans;
  }

  bool TakesLengths() const { return takes_lengths_; }
  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  void Init() {
    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
    // The first input is the features. A second one is each utterance's
    // count of valid (stacked) input frames, from an export that masks
    // padding; a second output is then the count of valid output frames.
    takes_lengths_ = input_names_.size() == 2;
    if (input_names_.size() > 2) {
      SHERPA_ONNX_LOGE(
          "A w2v-BERT CTC model takes input_features and optionally "
          "input_lengths. This one takes %d inputs.",
          static_cast<int32_t>(input_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }
    vocab_size_ = static_cast<int32_t>(
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2]);
  }

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
};

OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(Manager *mgr,
                                               const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

OfflineW2vBertCtcModel::~OfflineW2vBertCtcModel() = default;

std::vector<Ort::Value> OfflineW2vBertCtcModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

bool OfflineW2vBertCtcModel::SupportBatchProcessing() const {
  return impl_->TakesLengths();
}

int32_t OfflineW2vBertCtcModel::VocabSize() const { return impl_->VocabSize(); }

OrtAllocator *OfflineW2vBertCtcModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineW2vBertCtcModel::NormalizeFeatures(float *features,
                                               int32_t num_frames,
                                               int32_t feat_dim) const {
  NormalizeW2vBertFeatures(features, num_frames, feat_dim);
}

void NormalizeW2vBertFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim) {
  if (num_frames < 2) return;
  for (int32_t d = 0; d != feat_dim; ++d) {
    double sum = 0;
    for (int32_t t = 0; t != num_frames; ++t) sum += features[t * feat_dim + d];
    double mean = sum / num_frames;
    double squares = 0;
    for (int32_t t = 0; t != num_frames; ++t) {
      double x = features[t * feat_dim + d] - mean;
      squares += x * x;
    }
    // The sample variance, as the feature extractor computes it.
    double scale = 1.0 / std::sqrt(squares / (num_frames - 1) + 1e-7);
    for (int32_t t = 0; t != num_frames; ++t) {
      features[t * feat_dim + d] =
          static_cast<float>((features[t * feat_dim + d] - mean) * scale);
    }
  }
}

#if __ANDROID_API__ >= 9
template OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(
    AAssetManager *mgr, const OfflineModelConfig &config);
#endif

#if __OHOS__
template OfflineW2vBertCtcModel::OfflineW2vBertCtcModel(
    NativeResourceManager *mgr, const OfflineModelConfig &config);
#endif

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/offline-w2v-bert-ctc-model.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h"

#include <algorithm>
#include <array>
#include <memory>
#include <sstream>
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
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

void NormalizeW2vBertFeatures(float *features, int32_t num_frames,
                              int32_t feat_dim) {
  if (num_frames < 2) return;
  using RowMajorMat =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<RowMajorMat> x(features, num_frames, feat_dim);

  Eigen::RowVectorXf mean = x.colwise().mean();
  x.rowwise() -= mean;

  // The sample variance, as the feature extractor computes it.
  Eigen::Array<float, 1, Eigen::Dynamic> var =
      x.array().square().colwise().sum() / static_cast<float>(num_frames - 1);
  Eigen::Array<float, 1, Eigen::Dynamic> inv_std = (var + 1e-7f).rsqrt();
  x.array().rowwise() *= inv_std;
}

class OfflineW2vBertCtcModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    // From the path, not a buffer: the fp32 weights are in model.weights
    // beside model.onnx.
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.w2v_bert_ctc.model), sess_opts_);
    Init();
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflineModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    auto buf = ReadFile(mgr, config_.w2v_bert_ctc.model);
    sess_ = std::make_unique<Ort::Session>(env_, buf.data(), buf.size(),
                                           sess_opts_);
    Init();
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    std::vector<Ort::Value> out;
    if (takes_lengths_) {
      // The model masks the padding, whatever its value. With an int8 model
      // the batch still shares activation quantization ranges, so a batch
      // can differ slightly from single decoding; see the class comment.
      std::array<Ort::Value, 2> inputs = {std::move(features),
                                          std::move(features_length)};
      out =
          sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                     output_names_ptr_.data(), output_names_ptr_.size());
    } else {
      ZeroPadding(&features, features_length);
      out = sess_->Run({}, input_names_ptr_.data(), &features, 1,
                       output_names_ptr_.data(), output_names_ptr_.size());
      if (out.size() == 1) {
        out.push_back(OutputLengths(features_length, out[0]));
      }
    }

    if (!normalized_) {
      auto shape = out[0].GetTensorTypeAndShapeInfo().GetShape();
      LogSoftmax(out[0].GetTensorMutableData<float>(),
                 static_cast<int32_t>(shape[2]),
                 static_cast<int32_t>(shape[0] * shape[1]));
    }
    return out;
  }

  bool Batching() const { return batching_; }
  int32_t SubsamplingFactor() const { return subsampling_factor_; }
  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  void Init() {
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

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    // An export without model_type predates the metadata; one with another
    // type is the wrong model.
    std::string model_type;
    SHERPA_ONNX_READ_META_DATA_STR_ALLOW_EMPTY(model_type, "model_type");
    if (!model_type.empty() && model_type != "w2v-bert-ctc") {
      SHERPA_ONNX_LOGE(
          "Expected model type 'w2v-bert-ctc'. Given: '%s'. Please export the "
          "model with scripts/w2v-bert/export_onnx.py",
          model_type.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    // Version 2 normalizes in the graph; version 1, and an export without a
    // version, did not, and Forward normalizes theirs. Outputs are taken by
    // position, so either output naming (log_probs, or logits) loads.
    int32_t version = 0;
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(version, "version", 0);
    normalized_ = version >= 2;

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);
    // The first input is the features. A second one is each utterance's
    // valid rows, from an export that masks padding; a second output is the
    // valid output frames.
    takes_lengths_ = input_names_.size() == 2;
    if (input_names_.empty() || input_names_.size() > 2 ||
        output_names_.empty() || output_names_.size() > 2 ||
        (takes_lengths_ && output_names_.size() != 2)) {
      SHERPA_ONNX_LOGE(
          "A w2v-BERT CTC model takes input_features and input_lengths and "
          "returns log_probs and log_probs_lengths (or takes and returns the "
          "first alone). This one has %d inputs and %d outputs. Please export "
          "it with scripts/w2v-bert/export_onnx.py",
          static_cast<int32_t>(input_names_.size()),
          static_cast<int32_t>(output_names_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    vocab_size_ = static_cast<int32_t>(
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2]);
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(vocab_size_, "vocab_size",
                                            vocab_size_);
    // Filterbank frames per output frame: the frame pairs and any strided
    // adapter layers.
    SHERPA_ONNX_READ_META_DATA_WITH_DEFAULT(subsampling_factor_,
                                            "subsampling_factor", 2);
    if (subsampling_factor_ < 2 || subsampling_factor_ % 2 != 0) {
      SHERPA_ONNX_LOGE(
          "A w2v-BERT model's subsampling_factor must be a positive multiple "
          "of 2. Given: %d",
          subsampling_factor_);
      SHERPA_ONNX_EXIT(-1);
    }
    batching_ = BatchingSupported(meta_data, allocator, takes_lengths_);
  }

  // An export without input_lengths has no mask. Batches of it are padded
  // with zeros, the value of normalized features on average, as its
  // batching_supported check was run.
  static void ZeroPadding(Ort::Value *features,
                          const Ort::Value &features_length) {
    auto shape = features->GetTensorTypeAndShapeInfo().GetShape();
    const int64_t batch = shape[0], rows = shape[1], dim = shape[2];
    float *p = features->GetTensorMutableData<float>();
    const int64_t *length = features_length.GetTensorData<int64_t>();
    for (int64_t b = 0; b != batch; ++b) {
      std::fill(p + (b * rows + length[b]) * dim, p + (b + 1) * rows * dim,
                0.0f);
    }
  }

  // The valid output frames of an export that does not return them: its
  // strided adapter layers round the rows up.
  Ort::Value OutputLengths(const Ort::Value &features_length,
                           const Ort::Value &out) {
    const int64_t out_frames = out.GetTensorTypeAndShapeInfo().GetShape()[1];
    auto batch = static_cast<int64_t>(
        features_length.GetTensorTypeAndShapeInfo().GetShape()[0]);
    std::array<int64_t, 1> shape = {batch};
    Ort::Value lengths =
        Ort::Value::CreateTensor<int64_t>(allocator_, shape.data(), 1);
    const int64_t *rows = features_length.GetTensorData<int64_t>();
    int64_t *frames = lengths.GetTensorMutableData<int64_t>();
    const int64_t stride = subsampling_factor_ / 2;  // rows per output frame
    for (int64_t b = 0; b != batch; ++b) {
      frames[b] = std::min((rows[b] + stride - 1) / stride, out_frames);
    }
    return lengths;
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
  int32_t subsampling_factor_ = 2;
  bool normalized_ = true;     // version 2: log-softmax in the graph
  bool takes_lengths_ = true;  // input_lengths, and padding masked
  bool batching_ = true;       // BatchingSupported, by default takes_lengths_
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

int32_t OfflineW2vBertCtcModel::VocabSize() const { return impl_->VocabSize(); }

int32_t OfflineW2vBertCtcModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineW2vBertCtcModel::Allocator() const {
  return impl_->Allocator();
}

bool OfflineW2vBertCtcModel::SupportBatchProcessing() const {
  return impl_->Batching();
}

void OfflineW2vBertCtcModel::NormalizeFeatures(float *features,
                                               int32_t num_frames,
                                               int32_t feat_dim) const {
  NormalizeW2vBertFeatures(features, num_frames, feat_dim);
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

// sherpa-onnx/csrc/offline-w2v-bert-ctc-litert-model.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-litert-model.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-model.h"

// LiteRT's C API reports failure by status; none of these calls can be
// recovered from, so each failure names its call and ends the program.
#define SHERPA_ONNX_LITERT_OK(call)                                 \
  do {                                                              \
    LiteRtStatus litert_status = (call);                            \
    if (litert_status != kLiteRtStatusOk) {                         \
      SHERPA_ONNX_LOGE("LiteRT call failed with status %d: %s",     \
                       static_cast<int32_t>(litert_status), #call); \
      SHERPA_ONNX_EXIT(-1);                                         \
    }                                                               \
  } while (0)

namespace sherpa_onnx {

namespace {

// Two filterbank frames become one input frame.
constexpr int32_t kStack = 2;

// The graph's inputs and outputs by name; an enumerator is the name's
// position in its list.
constexpr const char *kInputNames[] = {"input_features", "input_lengths"};
constexpr const char *kOutputNames[] = {"logits", "logits_lengths"};
enum Input { kFeatures = 0, kLengths = 1 };
enum Output { kLogits = 0, kLogitsLengths = 1 };

// One input or output of the graph's signature, with the buffer it is run on.
struct Port {
  LiteRtTensorBuffer buffer = nullptr;
  LiteRtRankedTensorType type;
};

// Locks a tensor buffer for the lifetime of the object.
class Locked {
 public:
  Locked(LiteRtTensorBuffer buffer, LiteRtTensorBufferLockMode mode)
      : buffer_(buffer) {
    SHERPA_ONNX_LITERT_OK(LiteRtLockTensorBuffer(buffer_, &memory_, mode));
  }
  ~Locked() { LiteRtUnlockTensorBuffer(buffer_); }
  Locked(const Locked &) = delete;
  Locked &operator=(const Locked &) = delete;

  template <typename T>
  T *As() const {
    return static_cast<T *>(memory_);
  }

 private:
  LiteRtTensorBuffer buffer_;
  void *memory_ = nullptr;
};

}  // namespace

class OfflineW2vBertCtcLiteRtModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config) : allocator_{} {
    bool npu = config.provider == "npu";

    LiteRtEnvOption env_option;
    env_option.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
    env_option.value.type = kLiteRtAnyTypeString;
    env_option.value.str_value = config.litert_dispatch_dir.c_str();
    SHERPA_ONNX_LITERT_OK(LiteRtCreateEnvironment(
        config.litert_dispatch_dir.empty() ? 0 : 1, &env_option, &env_));

    SHERPA_ONNX_LITERT_OK(
        LiteRtCreateModelFromFile(env_, config.w2v_bert_ctc.c_str(), &model_));
    LiteRtOptions options;
    SHERPA_ONNX_LITERT_OK(LiteRtCreateOptions(&options));
    SHERPA_ONNX_LITERT_OK(LiteRtSetOptionsHardwareAccelerators(
        options, npu ? kLiteRtHwAcceleratorNpu : kLiteRtHwAcceleratorCpu));
    SHERPA_ONNX_LITERT_OK(
        LiteRtCreateCompiledModel(env_, model_, options, &compiled_));
    LiteRtDestroyOptions(options);

    LiteRtSignature signature;
    SHERPA_ONNX_LITERT_OK(LiteRtGetModelSignature(model_, 0, &signature));
    // Buffers are passed to a run in the signature's order, which need not
    // be the order of the names here.
    inputs_ = Ports(signature, true, kInputNames, &input_index_);
    outputs_ = Ports(signature, false, kOutputNames, &output_index_);

    const auto &features = In(kFeatures).type.layout;
    const auto &logits = Out(kLogits).type.layout;
    if (features.rank != 3 || features.dimensions[0] != 1 || logits.rank != 3 ||
        In(kLengths).type.element_type != kLiteRtElementTypeInt32 ||
        Out(kLogitsLengths).type.element_type != kLiteRtElementTypeInt32) {
      SHERPA_ONNX_LOGE(
          "%s is not a w2v-BERT CTC LiteRT graph: it must take input_features "
          "(1, P, 160) and input_lengths int32 (1) and return logits (1, F, "
          "vocab_size).",
          config.w2v_bert_ctc.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    max_frames_ = features.dimensions[1];
    feat_dim_ = features.dimensions[2];
    out_frames_ = logits.dimensions[1];
    vocab_size_ = logits.dimensions[2];

    if (config.debug) {
      bool accelerated = false;
      LiteRtCompiledModelIsFullyAccelerated(compiled_, &accelerated);
      SHERPA_ONNX_LOGE(
          "LiteRT w2v-BERT CTC: %d input frames (%.1f s), %d output frames, "
          "vocabulary %d, accelerator %s, fully accelerated: %d",
          max_frames_, max_frames_ * kStack * 0.01f, out_frames_, vocab_size_,
          npu ? "npu" : "cpu", static_cast<int32_t>(accelerated));
    }
  }

  ~Impl() {
    for (auto *ports : {&inputs_, &outputs_}) {
      for (auto &port : *ports) LiteRtDestroyTensorBuffer(port.buffer);
    }
    LiteRtDestroyCompiledModel(compiled_);
    LiteRtDestroyModel(model_);
    LiteRtDestroyEnvironment(env_);
  }

  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    auto shape = features.GetTensorTypeAndShapeInfo().GetShape();
    if (shape[0] != 1 || shape[2] * kStack != feat_dim_) {
      SHERPA_ONNX_LOGE(
          "A LiteRT w2v-BERT CTC graph takes one utterance of %d-dimensional "
          "features per call. Given: batch %d, dimension %d",
          feat_dim_ / kStack, static_cast<int32_t>(shape[0]),
          static_cast<int32_t>(shape[2]));
      SHERPA_ONNX_EXIT(-1);
    }
    // An odd last frame is dropped, as the feature extractor drops it.
    auto frames = static_cast<int32_t>(
        features_length.GetTensorData<int64_t>()[0] / kStack);
    if (frames > max_frames_) {
      SHERPA_ONNX_LOGE(
          "An utterance of %.1f s is longer than this LiteRT graph's %.1f s; "
          "only its first %.1f s are decoded. Cut the audio or export a "
          "longer graph.",
          frames * kStack * 0.01f, max_frames_ * kStack * 0.01f,
          max_frames_ * kStack * 0.01f);
      frames = max_frames_;
    }

    std::array<int64_t, 3> logits_shape = {1, out_frames_, vocab_size_};
    Ort::Value logits = Ort::Value::CreateTensor<float>(
        allocator_, logits_shape.data(), logits_shape.size());
    std::array<int64_t, 1> length_shape = {1};
    Ort::Value logits_length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());

    {
      // The buffers belong to the model, so one utterance runs at a time.
      std::lock_guard<std::mutex> lock(mutex_);
      {
        Locked in(In(kFeatures).buffer, kLiteRtTensorBufferLockModeWrite);
        size_t valid = static_cast<size_t>(frames) * feat_dim_;
        // Consecutive frames are already adjacent in memory: stacking is
        // the same values under another shape.
        std::copy_n(features.GetTensorData<float>(), valid, in.As<float>());
        std::fill(in.As<float>() + valid,
                  in.As<float>() + static_cast<size_t>(max_frames_) * feat_dim_,
                  0.0f);
      }
      {
        Locked in(In(kLengths).buffer, kLiteRtTensorBufferLockModeWrite);
        *in.As<int32_t>() = frames;
      }
      // In the signature's order, which is the order of inputs_, outputs_.
      std::array<LiteRtTensorBuffer, 2> in = {inputs_[0].buffer,
                                              inputs_[1].buffer};
      std::array<LiteRtTensorBuffer, 2> out = {outputs_[0].buffer,
                                               outputs_[1].buffer};
      SHERPA_ONNX_LITERT_OK(LiteRtRunCompiledModel(
          compiled_, 0, in.size(), in.data(), out.size(), out.data()));
      {
        Locked result(Out(kLogits).buffer, kLiteRtTensorBufferLockModeRead);
        std::copy_n(result.As<float>(),
                    static_cast<size_t>(out_frames_) * vocab_size_,
                    logits.GetTensorMutableData<float>());
      }
      {
        Locked result(Out(kLogitsLengths).buffer,
                      kLiteRtTensorBufferLockModeRead);
        logits_length.GetTensorMutableData<int64_t>()[0] =
            std::min(*result.As<int32_t>(), out_frames_);
      }
    }
    LogSoftmax(logits.GetTensorMutableData<float>(), vocab_size_, out_frames_);

    std::vector<Ort::Value> ans;
    ans.push_back(std::move(logits));
    ans.push_back(std::move(logits_length));
    return ans;
  }

  int32_t VocabSize() const { return vocab_size_; }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  // The signature's inputs or outputs in its own order, each with a buffer;
  // (*index)[k] is the position in that order of names[k].
  std::vector<Port> Ports(LiteRtSignature signature, bool input,
                          const char *const (&names)[2],
                          std::array<size_t, 2> *index) {
    LiteRtParamIndex n = 0;
    SHERPA_ONNX_LITERT_OK(input ? LiteRtGetNumSignatureInputs(signature, &n)
                                : LiteRtGetNumSignatureOutputs(signature, &n));
    if (n != std::size(names)) {
      SHERPA_ONNX_LOGE("Expected %d %s in the LiteRT graph, found %d",
                       static_cast<int32_t>(std::size(names)),
                       input ? "inputs" : "outputs", static_cast<int32_t>(n));
      SHERPA_ONNX_EXIT(-1);
    }
    std::vector<Port> ports(n);
    for (LiteRtParamIndex i = 0; i != n; ++i) {
      const char *name = nullptr;
      LiteRtTensor tensor;
      LiteRtTensorBufferRequirements requirements;
      if (input) {
        SHERPA_ONNX_LITERT_OK(LiteRtGetSignatureInputName(signature, i, &name));
        SHERPA_ONNX_LITERT_OK(
            LiteRtGetSignatureInputTensorByIndex(signature, i, &tensor));
        SHERPA_ONNX_LITERT_OK(LiteRtGetCompiledModelInputBufferRequirements(
            compiled_, 0, i, &requirements));
      } else {
        SHERPA_ONNX_LITERT_OK(
            LiteRtGetSignatureOutputName(signature, i, &name));
        SHERPA_ONNX_LITERT_OK(
            LiteRtGetSignatureOutputTensorByIndex(signature, i, &tensor));
        SHERPA_ONNX_LITERT_OK(LiteRtGetCompiledModelOutputBufferRequirements(
            compiled_, 0, i, &requirements));
      }
      auto found =
          std::find_if(std::begin(names), std::end(names),
                       [name](auto *s) { return std::strcmp(s, name) == 0; });
      if (found == std::end(names)) {
        SHERPA_ONNX_LOGE("Unexpected %s in the LiteRT graph: %s",
                         input ? "input" : "output", name);
        SHERPA_ONNX_EXIT(-1);
      }
      (*index)[found - std::begin(names)] = i;
      SHERPA_ONNX_LITERT_OK(LiteRtGetRankedTensorType(tensor, &ports[i].type));
      SHERPA_ONNX_LITERT_OK(LiteRtCreateManagedTensorBufferFromRequirements(
          env_, &ports[i].type, requirements, &ports[i].buffer));
    }
    return ports;
  }

  const Port &In(Input which) const { return inputs_[input_index_[which]]; }
  const Port &Out(Output which) const { return outputs_[output_index_[which]]; }

  Ort::AllocatorWithDefaultOptions allocator_;
  LiteRtEnvironment env_ = nullptr;
  LiteRtModel model_ = nullptr;
  LiteRtCompiledModel compiled_ = nullptr;

  std::vector<Port> inputs_;   // in the signature's order
  std::vector<Port> outputs_;  // in the signature's order
  // Position in inputs_ of each Input; in outputs_ of each Output.
  std::array<size_t, 2> input_index_ = {0, 1};
  std::array<size_t, 2> output_index_ = {0, 1};

  int32_t max_frames_ = 0;  // P, stacked input frames
  int32_t feat_dim_ = 0;    // 160
  int32_t out_frames_ = 0;  // F
  int32_t vocab_size_ = 0;
  std::mutex mutex_;
};

OfflineW2vBertCtcLiteRtModel::OfflineW2vBertCtcLiteRtModel(
    const OfflineModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineW2vBertCtcLiteRtModel::~OfflineW2vBertCtcLiteRtModel() = default;

std::vector<Ort::Value> OfflineW2vBertCtcLiteRtModel::Forward(
    Ort::Value features, Ort::Value features_length) {
  return impl_->Forward(std::move(features), std::move(features_length));
}

int32_t OfflineW2vBertCtcLiteRtModel::VocabSize() const {
  return impl_->VocabSize();
}

OrtAllocator *OfflineW2vBertCtcLiteRtModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineW2vBertCtcLiteRtModel::NormalizeFeatures(float *features,
                                                     int32_t num_frames,
                                                     int32_t feat_dim) const {
  NormalizeW2vBertFeatures(features, num_frames, feat_dim);
}

}  // namespace sherpa_onnx

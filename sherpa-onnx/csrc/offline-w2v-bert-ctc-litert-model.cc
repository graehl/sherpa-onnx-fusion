// sherpa-onnx/csrc/offline-w2v-bert-ctc-litert-model.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-w2v-bert-ctc-litert-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // NOLINT
#include <numeric>
#include <sstream>
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
#include "sherpa-onnx/csrc/text-utils.h"

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

// Two filterbank frames of 80 bins become one input row of 160 values.
constexpr int32_t kStack = 2;
constexpr float kRowSeconds = 0.02f;

// The graph's inputs and outputs by name; an enumerator is the name's
// position in its list.
constexpr const char *kInputNames[] = {"input_features", "input_lengths"};
constexpr const char *kOutputNames[] = {"logits", "logits_lengths"};
enum Input { kFeatures = 0, kLengths = 1 };
enum Output { kLogits = 0, kLogitsLengths = 1 };

// One input or output of a signature, with the buffer it is run on.
struct Port {
  LiteRtTensorBuffer buffer = nullptr;
  LiteRtRankedTensorType type;
  LiteRtTensorBufferRequirements requirements = nullptr;
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

// One fixed length of the model: a signature of a compiled .tflite file,
// with its own buffers. A file of several signatures holds its weights once.
class FixedLengthGraph {
 public:
  FixedLengthGraph(LiteRtEnvironment env, LiteRtModel model,
                   LiteRtCompiledModel compiled, LiteRtParamIndex signature,
                   const std::string &path)
      : compiled_(compiled), signature_(signature) {
    LiteRtSignature handle;
    SHERPA_ONNX_LITERT_OK(LiteRtGetModelSignature(model, signature, &handle));
    // Buffers are passed to a run in the signature's order, which need not
    // be the order of the names here.
    inputs_ = Ports(env, handle, true, kInputNames, &input_index_);
    outputs_ = Ports(env, handle, false, kOutputNames, &output_index_);
    const auto &features = In(kFeatures).type.layout;
    const auto &logits = Out(kLogits).type.layout;
    const auto is_length = [](const Port &port) {
      return port.type.element_type == kLiteRtElementTypeInt32 &&
             port.type.layout.rank == 1 && port.type.layout.dimensions[0] == 1;
    };
    if (features.rank != 3 || features.dimensions[0] != 1 || logits.rank != 3 ||
        features.dimensions[1] < 2 || features.dimensions[2] != 160 ||
        logits.dimensions[0] != 1 || logits.dimensions[1] <= 0 ||
        logits.dimensions[2] <= 0 ||
        In(kFeatures).type.element_type != kLiteRtElementTypeFloat32 ||
        Out(kLogits).type.element_type != kLiteRtElementTypeFloat32 ||
        !is_length(In(kLengths)) || !is_length(Out(kLogitsLengths))) {
      SHERPA_ONNX_LOGE(
          "%s is not a w2v-BERT CTC LiteRT graph: each signature must take "
          "input_features (1, P, 160) and input_lengths int32 (1) and return "
          "logits (1, F, vocab_size) and logits_lengths int32 (1).",
          path.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    positions_ = features.dimensions[1];
    feat_dim_ = features.dimensions[2];
    out_frames_ = logits.dimensions[1];
    vocab_size_ = logits.dimensions[2];
    output_stride_ = out_frames_ == positions_ ? 1 : 2;
    if (out_frames_ != (positions_ + output_stride_ - 1) / output_stride_) {
      SHERPA_ONNX_LOGE(
          "LiteRT w2v-BERT needs one output per one or two input "
          "rows: %s has %d input rows and %d output frames",
          path.c_str(), positions_, out_frames_);
      SHERPA_ONNX_EXIT(-1);
    }
    for (auto *ports : {&inputs_, &outputs_}) {
      for (auto &port : *ports) {
        if (port.type.layout.has_strides) {
          SHERPA_ONNX_LOGE("LiteRT w2v-BERT needs contiguous tensors: %s",
                           path.c_str());
          SHERPA_ONNX_EXIT(-1);
        }
        SHERPA_ONNX_LITERT_OK(LiteRtCreateManagedTensorBufferFromRequirements(
            env, &port.type, port.requirements, &port.buffer));
      }
    }
  }
  ~FixedLengthGraph() {
    for (auto *ports : {&inputs_, &outputs_}) {
      for (auto &port : *ports) LiteRtDestroyTensorBuffer(port.buffer);
    }
  }
  FixedLengthGraph(const FixedLengthGraph &) = delete;
  FixedLengthGraph &operator=(const FixedLengthGraph &) = delete;

  int32_t positions() const { return positions_; }
  int32_t feat_dim() const { return feat_dim_; }
  int32_t out_frames() const { return out_frames_; }
  int32_t vocab_size() const { return vocab_size_; }
  int32_t output_stride() const { return output_stride_; }

  // Runs `rows` (at most positions()) rows of normalized features, zero
  // padded; returns the valid output frames.
  int32_t Run(const float *features, int32_t rows) {
    {
      Locked in(In(kFeatures).buffer, kLiteRtTensorBufferLockModeWrite);
      size_t valid = static_cast<size_t>(rows) * feat_dim_;
      std::copy_n(features, valid, in.As<float>());
      std::fill(in.As<float>() + valid,
                in.As<float>() + static_cast<size_t>(positions_) * feat_dim_,
                0.0f);
    }
    {
      Locked in(In(kLengths).buffer, kLiteRtTensorBufferLockModeWrite);
      *in.As<int32_t>() = rows;
    }
    std::array<LiteRtTensorBuffer, 2> in = {inputs_[0].buffer,
                                            inputs_[1].buffer};
    std::array<LiteRtTensorBuffer, 2> out = {outputs_[0].buffer,
                                             outputs_[1].buffer};
    SHERPA_ONNX_LITERT_OK(LiteRtRunCompiledModel(
        compiled_, signature_, in.size(), in.data(), out.size(), out.data()));
    Locked lengths(Out(kLogitsLengths).buffer, kLiteRtTensorBufferLockModeRead);
    const int32_t frames = *lengths.As<int32_t>();
    if (frames != (rows + output_stride_ - 1) / output_stride_) {
      SHERPA_ONNX_LOGE(
          "LiteRT logits_lengths %d disagrees with %d input rows "
          "and output stride %d",
          frames, rows, output_stride_);
      SHERPA_ONNX_EXIT(-1);
    }
    return frames;
  }

  // Adds weight(f) times the last run's logits of each of its first `frames`
  // frames f into sum's rows from row `offset`, and weight(f) into weights.
  template <typename Weight>
  void AccumulateLogits(int32_t frames, int32_t offset, Weight weight,
                        std::vector<double> *sum,
                        std::vector<double> *weights) const {
    Locked logits(Out(kLogits).buffer, kLiteRtTensorBufferLockModeRead);
    const float *row = logits.As<float>();
    for (int32_t f = 0; f < frames; ++f, row += vocab_size_) {
      double w = weight(f);
      double *to = sum->data() + static_cast<size_t>(offset + f) * vocab_size_;
      for (int32_t v = 0; v < vocab_size_; ++v) to[v] += w * row[v];
      (*weights)[offset + f] += w;
    }
  }

 private:
  std::vector<Port> Ports(LiteRtEnvironment env, LiteRtSignature signature,
                          bool input, const char *const (&names)[2],
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
      if (input) {
        SHERPA_ONNX_LITERT_OK(LiteRtGetSignatureInputName(signature, i, &name));
        SHERPA_ONNX_LITERT_OK(
            LiteRtGetSignatureInputTensorByIndex(signature, i, &tensor));
        SHERPA_ONNX_LITERT_OK(LiteRtGetCompiledModelInputBufferRequirements(
            compiled_, signature_, i, &ports[i].requirements));
      } else {
        SHERPA_ONNX_LITERT_OK(
            LiteRtGetSignatureOutputName(signature, i, &name));
        SHERPA_ONNX_LITERT_OK(
            LiteRtGetSignatureOutputTensorByIndex(signature, i, &tensor));
        SHERPA_ONNX_LITERT_OK(LiteRtGetCompiledModelOutputBufferRequirements(
            compiled_, signature_, i, &ports[i].requirements));
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
    }
    return ports;
  }

  const Port &In(Input which) const { return inputs_[input_index_[which]]; }
  const Port &Out(Output which) const { return outputs_[output_index_[which]]; }

  LiteRtCompiledModel compiled_;
  LiteRtParamIndex signature_;
  std::vector<Port> inputs_;   // in the signature's order
  std::vector<Port> outputs_;  // in the signature's order
  // Position in inputs_ of each Input; in outputs_ of each Output.
  std::array<size_t, 2> input_index_ = {0, 1};
  std::array<size_t, 2> output_index_ = {0, 1};
  int32_t positions_ = 0;  // P, input rows
  int32_t feat_dim_ = 0;   // 160
  int32_t out_frames_ = 0;
  int32_t vocab_size_ = 0;
  int32_t output_stride_ = 1;
};

// How a clip longer than one window is run: overlapping windows of one graph,
// in input rows of 20 ms.
struct WindowOptions {
  int32_t window_rows = 0;  // 0: the longest graph
  int32_t min_overlap_rows = 150;
  int32_t slack_rows = 50;
};

// --litert-window W:O:S, in seconds.
WindowOptions ParseWindowOptions(const std::string &spec) {
  WindowOptions o;
  if (spec.empty()) return o;
  std::vector<float> seconds;
  std::istringstream in(spec);
  for (std::string part; std::getline(in, part, ':');) {
    std::istringstream number(part);
    double value;
    if (!(number >> value) || !(number >> std::ws).eof() ||
        !std::isfinite(value) || value < 0 ||
        value / kRowSeconds > std::numeric_limits<int32_t>::max() / 2) {
      SHERPA_ONNX_LOGE("--litert-window needs finite nonnegative seconds: %s",
                       spec.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    seconds.push_back(value);
  }
  if (seconds.size() != 3 || spec.back() == ':') {
    SHERPA_ONNX_LOGE("--litert-window is WINDOW_S:OVERLAP_S:SLACK_S, not %s",
                     spec.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  o.window_rows = static_cast<int32_t>(std::lround(seconds[0] / kRowSeconds));
  o.min_overlap_rows =
      static_cast<int32_t>(std::lround(seconds[1] / kRowSeconds));
  o.slack_rows = static_cast<int32_t>(std::lround(seconds[2] / kRowSeconds));
  if (o.window_rows < 2) {
    SHERPA_ONNX_LOGE(
        "--litert-window needs a positive window of at least 0.04 s");
    SHERPA_ONNX_EXIT(-1);
  }
  return o;
}

// Window starts aligned to output frames
// covering `rows` with windows of `window` rows: as few as keep neighbours
// overlapping by at least o.min_overlap_rows, evenly spaced, the first at 0
// and the last ending at the clip's end; then each interior window moves by up
// to o.slack_rows to the start whose two edge rows are quietest, keeping half
// the minimum overlap with both neighbours.
std::vector<int32_t> EvenWindowStarts(int32_t rows, int32_t window,
                                      const WindowOptions &o,
                                      const std::vector<float> &loudness,
                                      int32_t stride) {
  if (rows <= window) return {0};
  const int32_t step = window - o.min_overlap_rows;
  if (step < stride) {
    SHERPA_ONNX_LOGE("--litert-window overlap leaves no step");
    SHERPA_ONNX_EXIT(-1);
  }
  const int32_t count = 1 + (rows - window + step - 1) / step;
  int32_t last = rows - window;
  last += (stride - last % stride) % stride;
  std::vector<int32_t> starts(count);
  for (int32_t i = 0; i < count; ++i) {
    starts[i] = count == 1 ? 0
                           : static_cast<int32_t>(static_cast<int64_t>(last) *
                                                  i / (count - 1));
    starts[i] -= starts[i] % stride;
  }
  starts.back() = last;
  auto edges = [&](int32_t s) {
    return loudness[s] + loudness[std::min(s + window, rows) - 1];
  };
  const int32_t keep = o.min_overlap_rows / 2;
  for (int32_t i = 1; i + 1 < count; ++i) {
    int32_t lo = std::max({starts[i] - o.slack_rows, starts[i - 1] + stride,
                           starts[i + 1] - window + keep});
    int32_t hi = std::min({starts[i] + o.slack_rows, starts[i + 1] - stride,
                           starts[i - 1] + window - keep});
    lo += (stride - lo % stride) % stride;
    int32_t chosen = starts[i];
    for (int32_t s = lo; s <= hi; s += stride) {
      if (edges(s) < edges(chosen)) chosen = s;
    }
    starts[i] = chosen;
  }
  return starts;
}

}  // namespace

class OfflineW2vBertCtcLiteRtModel::Impl {
 public:
  explicit Impl(const OfflineModelConfig &config)
      : allocator_{}, windows_(ParseWindowOptions(config.litert_window)) {
    bool npu = config.provider == "npu";
    LiteRtEnvOption env_option;
    env_option.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
    env_option.value.type = kLiteRtAnyTypeString;
    env_option.value.str_value = config.litert_dispatch_dir.c_str();
    SHERPA_ONNX_LITERT_OK(LiteRtCreateEnvironment(
        config.litert_dispatch_dir.empty() ? 0 : 1, &env_option, &env_));

    std::vector<std::string> paths;
    SplitStringToVector(config.w2v_bert_ctc, ",", true, &paths);
    for (const auto &path : paths) {
      LiteRtModel model;
      SHERPA_ONNX_LITERT_OK(
          LiteRtCreateModelFromFile(env_, path.c_str(), &model));
      LiteRtOptions options;
      SHERPA_ONNX_LITERT_OK(LiteRtCreateOptions(&options));
      SHERPA_ONNX_LITERT_OK(LiteRtSetOptionsHardwareAccelerators(
          options, npu ? kLiteRtHwAcceleratorNpu : kLiteRtHwAcceleratorCpu));
      LiteRtCompiledModel compiled;
      SHERPA_ONNX_LITERT_OK(
          LiteRtCreateCompiledModel(env_, model, options, &compiled));
      LiteRtDestroyOptions(options);
      models_.push_back(model);
      compiled_.push_back(compiled);
      LiteRtParamIndex signatures = 0;
      SHERPA_ONNX_LITERT_OK(LiteRtGetNumModelSignatures(model, &signatures));
      for (LiteRtParamIndex s = 0; s < signatures; ++s) {
        graphs_.push_back(
            std::make_unique<FixedLengthGraph>(env_, model, compiled, s, path));
      }
      if (config.debug) {
        bool accelerated = false;
        LiteRtCompiledModelIsFullyAccelerated(compiled, &accelerated);
        SHERPA_ONNX_LOGE(
            "LiteRT %s: %d signature(s), accelerator %s, fully "
            "accelerated: %d",
            path.c_str(), static_cast<int32_t>(signatures), npu ? "npu" : "cpu",
            static_cast<int32_t>(accelerated));
      }
    }
    if (graphs_.empty()) {
      SHERPA_ONNX_LOGE("LiteRT model has no signatures");
      SHERPA_ONNX_EXIT(-1);
    }
    std::sort(graphs_.begin(), graphs_.end(),
              [](auto &a, auto &b) { return a->positions() < b->positions(); });
    for (const auto &graph : graphs_) {
      if (graph->vocab_size() != graphs_[0]->vocab_size() ||
          graph->feat_dim() != graphs_[0]->feat_dim() ||
          graph->output_stride() != graphs_[0]->output_stride()) {
        SHERPA_ONNX_LOGE(
            "The LiteRT graphs differ in vocabulary, features "
            "or output stride");
        SHERPA_ONNX_EXIT(-1);
      }
      if (config.debug) {
        SHERPA_ONNX_LOGE(
            "LiteRT length: %d input rows (%.1f s), %d output "
            "frames, vocabulary %d",
            graph->positions(), graph->positions() * kRowSeconds,
            graph->out_frames(), graph->vocab_size());
      }
    }
    window_graph_ = graphs_.back().get();
    if (windows_.window_rows > 0) {
      auto fits = std::find_if(graphs_.begin(), graphs_.end(), [&](auto &g) {
        return g->positions() >= windows_.window_rows;
      });
      if (fits == graphs_.end()) {
        SHERPA_ONNX_LOGE("--litert-window exceeds every supplied graph length");
        SHERPA_ONNX_EXIT(-1);
      }
      window_graph_ = fits->get();
    }
    if (windows_.min_overlap_rows >
        window_graph_->positions() - window_graph_->output_stride()) {
      SHERPA_ONNX_LOGE(
          "--litert-window overlap leaves no step for the "
          "selected graph length");
      SHERPA_ONNX_EXIT(-1);
    }
  }

  ~Impl() {
    graphs_.clear();
    for (auto compiled : compiled_) LiteRtDestroyCompiledModel(compiled);
    for (auto model : models_) LiteRtDestroyModel(model);
    LiteRtDestroyEnvironment(env_);
  }

  // features: (1, T, 80) filterbank frames as the recognizer computed them,
  // not yet normalized (NormalizeFeatures leaves them for this).
  std::vector<Ort::Value> Forward(Ort::Value features,
                                  Ort::Value features_length) {
    auto shape = features.GetTensorTypeAndShapeInfo().GetShape();
    const int32_t bins = graphs_[0]->feat_dim() / kStack;
    if (shape[0] != 1 || shape[2] != bins) {
      SHERPA_ONNX_LOGE(
          "A LiteRT w2v-BERT CTC graph takes one utterance of %d-dimensional "
          "features per call. Given: batch %d, dimension %d",
          bins, static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[2]));
      SHERPA_ONNX_EXIT(-1);
    }
    const auto num_frames =
        static_cast<int32_t>(features_length.GetTensorData<int64_t>()[0]);
    const float *raw = features.GetTensorData<float>();
    // An odd last frame is dropped, as the feature extractor's attention mask
    // drops it; the normalization still counts it.
    const int32_t rows = num_frames / kStack;
    const int32_t stride = graphs_[0]->output_stride();
    const int32_t total_frames = (rows + stride - 1) / stride;
    const int32_t vocab = graphs_[0]->vocab_size();

    auto fits = std::find_if(graphs_.begin(), graphs_.end(), [rows](auto &g) {
      return g->positions() >= rows;
    });
    FixedLengthGraph &graph =
        (fits != graphs_.end() &&
         (*fits)->positions() <= window_graph_->positions())
            ? **fits
            : *window_graph_;

    // Per row, the mean log filterbank energy: low means quiet.
    std::vector<float> loudness(rows);
    for (int32_t r = 0; r < rows; ++r) {
      const float *row = raw + static_cast<size_t>(r) * kStack * bins;
      loudness[r] = std::accumulate(row, row + kStack * bins, 0.0f);
    }
    std::vector<int32_t> starts =
        EvenWindowStarts(rows, graph.positions(), windows_, loudness, stride);

    std::vector<double> sum(static_cast<size_t>(total_frames) * vocab, 0.0);
    std::vector<double> weights(total_frames, 0.0);
    {
      // The buffers belong to the model, so one utterance runs at a time.
      std::lock_guard<std::mutex> lock(mutex_);
      for (int32_t start : starts) {
        const int32_t valid = std::min(graph.positions(), rows - start);
        // Each window normalized over its own frames, so that it looks like
        // an utterance of its own; the last also counts an odd last frame.
        const int32_t first = start * kStack;
        const int32_t end =
            start + valid == rows ? num_frames : (start + valid) * kStack;
        std::vector<float> window(raw + static_cast<size_t>(first) * bins,
                                  raw + static_cast<size_t>(end) * bins);
        NormalizeW2vBertFeatures(window.data(), end - first, bins);
        const int32_t frames = std::min(graph.Run(window.data(), valid),
                                        total_frames - start / stride);
        graph.AccumulateLogits(
            frames, start / stride,
            [frames](int32_t f) { return 1.0 + std::min(f, frames - 1 - f); },
            &sum, &weights);
      }
    }

    std::array<int64_t, 3> logits_shape = {1, total_frames, vocab};
    Ort::Value logits = Ort::Value::CreateTensor<float>(
        allocator_, logits_shape.data(), logits_shape.size());
    float *out = logits.GetTensorMutableData<float>();
    for (int32_t f = 0; f < total_frames; ++f) {
      for (int32_t v = 0; v < vocab; ++v) {
        const size_t i = static_cast<size_t>(f) * vocab + v;
        out[i] = static_cast<float>(sum[i] / weights[f]);
      }
    }
    LogSoftmax(out, vocab, total_frames);
    std::array<int64_t, 1> length_shape = {1};
    Ort::Value logits_length = Ort::Value::CreateTensor<int64_t>(
        allocator_, length_shape.data(), length_shape.size());
    logits_length.GetTensorMutableData<int64_t>()[0] = total_frames;

    std::vector<Ort::Value> ans;
    ans.push_back(std::move(logits));
    ans.push_back(std::move(logits_length));
    return ans;
  }

  int32_t VocabSize() const { return graphs_[0]->vocab_size(); }
  int32_t SubsamplingFactor() const {
    return kStack * graphs_[0]->output_stride();
  }
  OrtAllocator *Allocator() { return allocator_; }

 private:
  Ort::AllocatorWithDefaultOptions allocator_;
  WindowOptions windows_;
  LiteRtEnvironment env_ = nullptr;
  std::vector<LiteRtModel> models_;
  std::vector<LiteRtCompiledModel> compiled_;
  // Every signature of every file, shortest first.
  std::vector<std::unique_ptr<FixedLengthGraph>> graphs_;
  // Runs clips longer than every graph shorter than it.
  FixedLengthGraph *window_graph_ = nullptr;
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

int32_t OfflineW2vBertCtcLiteRtModel::SubsamplingFactor() const {
  return impl_->SubsamplingFactor();
}

OrtAllocator *OfflineW2vBertCtcLiteRtModel::Allocator() const {
  return impl_->Allocator();
}

void OfflineW2vBertCtcLiteRtModel::NormalizeFeatures(float *, int32_t,
                                                     int32_t) const {}

}  // namespace sherpa_onnx

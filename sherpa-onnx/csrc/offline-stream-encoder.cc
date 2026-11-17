// sherpa-onnx/csrc/offline-stream-encoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-stream-encoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

namespace {

// Output frames a window's output count may differ from its input length
// over the subsampling factor, for frames a front end loses at its edges.
constexpr int32_t kEdgeMarginFrames = 4;

// One encoder input: an utterance, or one window of it.
struct Segment {
  int32_t stream;  // index into the streams being encoded
  int32_t start;   // first feature frame in the utterance
  std::vector<float> features;
  int32_t num_frames;
};

}  // namespace

void MoveStreamNormalizationToEncoder(const OfflineEncoderConfig &config,
                                      FeatureExtractorConfig *feat_config,
                                      OfflineEncoderAdapter *adapter) {
  if (!config.window.Enabled() || feat_config->nemo_normalize_type.empty()) {
    return;
  }
  if (feat_config->nemo_normalize_type != "per_feature") {
    SHERPA_ONNX_LOGE("Only normalize_type=per_feature is implemented. Given %s",
                     feat_config->nemo_normalize_type.c_str());
    SHERPA_ONNX_EXIT(-1);
  }
  // The type also selects NeMo's log guard in the filterbank, so it stays.
  feat_config->nemo_normalize_in_stream = false;
  auto model = std::move(adapter->normalize);
  adapter->normalize = [model](float *f, int32_t num_frames, int32_t dim) {
    NemoNormalizePerFeature(f, num_frames, dim);
    if (model) model(f, num_frames, dim);
  };
}

OfflineStreamEncoder::OfflineStreamEncoder(OfflineEncoderAdapter adapter,
                                           const OfflineEncoderConfig &config,
                                           float frame_shift_ms)
    : adapter_(std::move(adapter)), config_(config.window) {
  const float frame_seconds = frame_shift_ms / 1000.0f;
  auto frames = [frame_seconds](float seconds) {
    return std::max(1,
                    static_cast<int32_t>(std::lround(seconds / frame_seconds)));
  };
  std::vector<float> lengths;
  config.ParseLengths(&lengths);  // Validate has checked it
  for (float seconds : lengths) lengths_.push_back(frames(seconds));
  if (config.pad_to > 0) pad_to_ = frames(config.pad_to);
  if ((!lengths_.empty() || pad_to_ > 0) &&
      (!adapter_.batch || !adapter_.pads_past_longest)) {
    SHERPA_ONNX_LOGE(
        "--encoder.lengths and --encoder.pad-to need a model whose encoder "
        "masks padding past the longest input of a batch; this one %s",
        adapter_.batch ? "cannot be padded past its longest input"
                       : "takes one unpadded utterance per call");
    SHERPA_ONNX_EXIT(-1);
  }
  if (!config_.Enabled()) return;
  if (adapter_.rows == WindowBlend::Rows::kHidden && config_.blend == "prob") {
    SHERPA_ONNX_LOGE(
        "--encoder.window.blend=prob needs a CTC model: this model's encoder "
        "output is not a distribution");
    SHERPA_ONNX_EXIT(-1);
  }
  schedule_ = ScheduleInUnits(config_, frame_seconds);
  schedule_.window = PaddedLength(schedule_.window);
  if (schedule_.window - schedule_.min_overlap < adapter_.subsampling_factor) {
    SHERPA_ONNX_LOGE(
        "--encoder.window.overlap leaves less than one output frame between "
        "window starts");
    SHERPA_ONNX_EXIT(-1);
  }
  // A front end loses output frames at a window's edges; the frame check in
  // Encode allows four, so an overlap shorter than that could leave frames
  // uncovered at some utterance lengths and stop the program mid-run.
  const int32_t least = kEdgeMarginFrames * adapter_.subsampling_factor;
  if (schedule_.min_overlap < least) {
    SHERPA_ONNX_LOGE(
        "--encoder.window.overlap must be at least %d feature frames (%.2f s) "
        "for this model, four output frames",
        least, least * frame_seconds);
    SHERPA_ONNX_EXIT(-1);
  }
}

int32_t OfflineStreamEncoder::PaddedLength(int32_t num_frames) const {
  auto fits = std::lower_bound(lengths_.begin(), lengths_.end(), num_frames);
  if (fits != lengths_.end()) return *fits;
  if (pad_to_ > 0) return (num_frames + pad_to_ - 1) / pad_to_ * pad_to_;
  return num_frames;
}

void OfflineStreamEncoder::Encode(OfflineStream **ss, int32_t n) const {
  std::vector<OfflineStream *> todo;
  for (int32_t i = 0; i != n; ++i) {
    if (!ss[i]->HasEncoderOutput()) todo.push_back(ss[i]);
  }
  if (todo.empty()) return;

  const int32_t feat_dim = todo[0]->FeatureDim();
  const int32_t grid = adapter_.subsampling_factor;
  const bool per_utterance = config_.normalize == "utterance";
  auto Normalize = [this](float *f, int32_t num_frames, int32_t dim) {
    if (adapter_.normalize) adapter_.normalize(f, num_frames, dim);
  };
  std::vector<Segment> segments;
  for (int32_t i = 0; i != static_cast<int32_t>(todo.size()); ++i) {
    std::vector<float> f = todo[i]->GetFrames();
    const auto num_frames = static_cast<int32_t>(f.size() / feat_dim);
    std::vector<int32_t> starts = {0};
    if (Windowed() && num_frames > schedule_.window) {
      // Per frame, the sum of its features: low means quiet.
      std::vector<float> loudness(num_frames);
      for (int32_t t = 0; t != num_frames; ++t) {
        const float *row = f.data() + static_cast<size_t>(t) * feat_dim;
        loudness[t] = std::accumulate(row, row + feat_dim, 0.0f);
      }
      // Neighbours must share the front end's edge loss and both windows'
      // context-only frames, or some output frame is kept by neither.
      const int32_t context = std::max(
          config_.context_frames,
          static_cast<int32_t>(schedule_.window / grid *
                               static_cast<double>(config_.context_fraction)));
      starts =
          EvenWindowStarts(num_frames, schedule_.window, schedule_, loudness,
                           grid, (kEdgeMarginFrames + 2 * context) * grid);
    }
    if (starts.size() == 1) {
      Normalize(f.data(), num_frames, feat_dim);
      segments.push_back({i, 0, std::move(f), num_frames});
      continue;
    }
    if (per_utterance) Normalize(f.data(), num_frames, feat_dim);
    for (int32_t start : starts) {
      const int32_t frames = std::min(schedule_.window, num_frames - start);
      const float *begin = f.data() + static_cast<size_t>(start) * feat_dim;
      Segment s{i, start,
                std::vector<float>(
                    begin, begin + static_cast<size_t>(frames) * feat_dim),
                frames};
      if (!per_utterance) Normalize(s.features.data(), frames, feat_dim);
      segments.push_back(std::move(s));
    }
  }

  // Each segment's encoder call and its item in that call.
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<std::shared_ptr<const EncoderOutputBatch>> outputs;
  std::vector<std::pair<int32_t, int32_t>> where(segments.size());
  // Runs segments [first, end) in one call, padded to the allowed length
  // that holds their longest.
  auto run = [&](size_t first, size_t end) {
    int32_t max_frames = 0;
    std::vector<int64_t> lengths;
    for (size_t k = first; k != end; ++k) {
      max_frames = std::max(max_frames, segments[k].num_frames);
      lengths.push_back(segments[k].num_frames);
    }
    max_frames = PaddedLength(max_frames);
    const auto batch = static_cast<int64_t>(end - first);
    std::array<int64_t, 3> shape = {batch, max_frames, feat_dim};
    Ort::Value x{nullptr};
    if (batch == 1 && max_frames == segments[first].num_frames) {
      x = Ort::Value::CreateTensor(memory_info, segments[first].features.data(),
                                   segments[first].features.size(),
                                   shape.data(), shape.size());
    } else {
      x = Ort::Value::CreateTensor<float>(adapter_.allocator, shape.data(),
                                          shape.size());
      float *p = x.GetTensorMutableData<float>();
      const size_t stride = static_cast<size_t>(max_frames) * feat_dim;
      std::fill(p, p + batch * stride, adapter_.padding_value);
      for (size_t k = first; k != end; ++k, p += stride) {
        std::copy(segments[k].features.begin(), segments[k].features.end(), p);
      }
    }
    std::array<int64_t, 1> length_shape = {batch};
    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, lengths.data(), lengths.size(),
                                 length_shape.data(), length_shape.size());
    for (size_t k = first; k != end; ++k) {
      where[k] = {static_cast<int32_t>(outputs.size()),
                  static_cast<int32_t>(k - first)};
    }
    auto out = adapter_.encode(std::move(x), std::move(x_length));
    outputs.push_back(
        std::make_shared<EncoderOutputBatch>(std::move(out[0]), out[1]));
  };
  if (adapter_.batch) {
    run(0, segments.size());
  } else {
    for (size_t k = 0; k != segments.size(); ++k) run(k, k + 1);
  }

  for (size_t first = 0; first != segments.size();) {
    size_t end = first + 1;
    while (end != segments.size() &&
           segments[end].stream == segments[first].stream) {
      ++end;
    }
    OfflineStream *stream = todo[segments[first].stream];
    const int32_t dim = outputs[where[first].first]->Dim();
    if (end == first + 1) {
      // A whole utterance: the stream shares its call's output.
      stream->SetEncoderOutput(outputs[where[first].first],
                               where[first].second);
      first = end;
      continue;
    }
    int32_t total = 0;
    for (size_t k = first; k != end; ++k) {
      const int32_t frames = outputs[where[k].first]->Frames(where[k].second);
      // A window's output must be on the grid its start is aligned to; the
      // margin allows for frames a convolutional front end loses at edges.
      if (std::abs(frames * grid - segments[k].num_frames) >
          kEdgeMarginFrames * grid) {
        SHERPA_ONNX_LOGE(
            "An encoder window of %d frames gave %d output frames, which does "
            "not match the model's subsampling factor %d",
            segments[k].num_frames, frames, grid);
        SHERPA_ONNX_EXIT(-1);
      }
      total = std::max(total, segments[k].start / grid + frames);
    }
    WindowBlend blend(total, dim, adapter_.rows, config_);
    for (size_t k = first; k != end; ++k) {
      const EncoderOutputBatch &out = *outputs[where[k].first];
      blend.Add(out.Row(where[k].second), out.Frames(where[k].second),
                segments[k].start / grid);
    }
    std::vector<float> values(static_cast<size_t>(total) * dim);
    if (!blend.Finish(values.data())) {
      SHERPA_ONNX_LOGE(
          "Encoder windows leave an output frame uncovered; increase "
          "--encoder.window.overlap");
      SHERPA_ONNX_EXIT(-1);
    }
    stream->SetEncoderOutput(
        std::make_shared<EncoderOutputBatch>(std::move(values), total, dim), 0);
    first = end;
  }
}

bool AnyHasEncoderOutput(OfflineStream **ss, int32_t n) {
  return std::any_of(ss, ss + n,
                     [](OfflineStream *s) { return s->HasEncoderOutput(); });
}

TakenEncoderOutputs TakeEncoderOutputs(OfflineStream **ss, int32_t n,
                                       OrtAllocator *allocator) {
  TakenEncoderOutputs ans;
  std::vector<std::shared_ptr<const EncoderOutputBatch>> batch(n);
  std::vector<int32_t> item(n);
  std::vector<int64_t> lengths(n);
  bool one_run = n > 0;  // consecutive items of one batch, in order
  for (int32_t i = 0; i != n; ++i) {
    batch[i] = ss[i]->TakeEncoderOutput(&item[i]);
    lengths[i] = batch[i]->Frames(item[i]);
    one_run = one_run && batch[i] == batch[0] && item[i] == item[0] + i;
  }

  std::array<int64_t, 1> length_shape = {n};
  ans.lengths = Ort::Value::CreateTensor<int64_t>(
      allocator, length_shape.data(), length_shape.size());
  std::copy(lengths.begin(), lengths.end(),
            ans.lengths.GetTensorMutableData<int64_t>());
  if (n == 0) return ans;

  const int64_t dim = batch[0]->Dim();
  if (one_run) {
    // A view of the batch's own rows: the decoders read only each item's
    // valid frames.
    const EncoderOutputBatch &b = *batch[0];
    std::array<int64_t, 3> shape = {n, b.MaxFrames(), dim};
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    ans.values = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(b.Row(item[0])),
        static_cast<size_t>(n) * b.MaxFrames() * dim, shape.data(),
        shape.size());
    ans.batches.push_back(std::move(batch[0]));
    return ans;
  }

  const int64_t max_frames = *std::max_element(lengths.begin(), lengths.end());
  std::array<int64_t, 3> shape = {n, max_frames, dim};
  ans.values =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = ans.values.GetTensorMutableData<float>();
  const size_t stride = static_cast<size_t>(max_frames) * dim;
  std::fill(p, p + n * stride, 0.0f);
  for (int32_t i = 0; i != n; ++i) {
    const float *row = batch[i]->Row(item[i]);
    std::copy(row, row + lengths[i] * dim, p + i * stride);
  }
  return ans;
}

}  // namespace sherpa_onnx

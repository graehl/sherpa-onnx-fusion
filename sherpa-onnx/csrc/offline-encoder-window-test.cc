// sherpa-onnx/csrc/offline-encoder-window-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-encoder-window.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/offline-stream-encoder.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

TEST(EvenWindowStarts, CoverTheInputOnTheOutputGrid) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uniform(0, 1);
  WindowSchedule s{1000, 300, 100};
  for (int32_t length : {1000, 1001, 2500, 7777}) {
    for (int32_t grid : {1, 4, 8}) {
      std::vector<float> loudness(length);
      for (auto &x : loudness) x = uniform(rng);
      auto starts = EvenWindowStarts(length, s.window, s, loudness, grid);
      ASSERT_FALSE(starts.empty());
      EXPECT_EQ(starts.front(), 0);
      EXPECT_GE(starts.back() + s.window, length);
      EXPECT_LT(starts.back(), length);
      for (size_t i = 0; i != starts.size(); ++i) {
        EXPECT_EQ(starts[i] % grid, 0);
        if (i > 0) {
          EXPECT_GT(starts[i], starts[i - 1]);
          EXPECT_GE(starts[i - 1] + s.window - starts[i], s.min_overlap / 2);
        }
      }
    }
  }
}

TEST(EvenWindowStarts, OneWindowForAShortInput) {
  std::vector<float> loudness(500, 0);
  EXPECT_EQ(EvenWindowStarts(500, 1000, {1000, 300, 100}, loudness, 4),
            std::vector<int32_t>{0});
}

namespace {

OfflineEncoderWindowConfig WindowConfig(const char *weight, const char *blend) {
  OfflineEncoderWindowConfig c;
  c.seconds = 1;
  c.weight = weight;
  c.blend = blend;
  return c;
}

}  // namespace

TEST(WindowBlend, AveragesAgreeingWindowsToTheirValue) {
  // Two windows of 4 frames overlapping by 2, every row the same
  // distribution; each mode must return it.
  const std::vector<float> row = {std::log(0.5f), std::log(0.3f),
                                  std::log(0.2f)};
  std::vector<float> window;
  for (int32_t f = 0; f != 4; ++f) {
    window.insert(window.end(), row.begin(), row.end());
  }
  for (auto [weight, blend] :
       {std::pair{"center", "log"}, {"center", "prob"}, {"cut", "log"}}) {
    WindowBlend b(6, 3, WindowBlend::Rows::kScores,
                  WindowConfig(weight, blend));
    b.Add(window.data(), 4, 0);
    b.Add(window.data(), 4, 2);
    std::vector<float> out(18);
    ASSERT_TRUE(b.Finish(out.data()));
    for (int32_t i = 0; i != 18; ++i) {
      EXPECT_NEAR(out[i], row[i % 3], 1e-6) << weight << " " << blend;
    }
  }
}

TEST(WindowBlend, CutTakesTheMoreCentralWindow) {
  // Window a covers frames 0..3, b frames 2..5. Frame 2 is a's frame 2
  // (weight 2) and b's frame 0 (weight 1); frame 3 is a's last (1) and b's
  // frame 1 (2).
  std::vector<float> a = {0, 0, 0, 0}, b = {1, 1, 1, 1};
  WindowBlend blend(6, 1, WindowBlend::Rows::kHidden,
                    WindowConfig("cut", "log"));
  blend.Add(a.data(), 4, 0);
  blend.Add(b.data(), 4, 2);
  std::vector<float> out(6);
  ASSERT_TRUE(blend.Finish(out.data()));
  EXPECT_EQ(out, (std::vector<float>{0, 0, 0, 1, 1, 1}));
}

TEST(WindowBlend, LogAndProbAverageInTheirOwnSpaces) {
  // One frame from two one-frame windows, p = (0.9, 0.1) and (0.5, 0.5),
  // equally weighted: prob gives the mean (0.7, 0.3); log gives the
  // renormalized geometric mean, (0.75, 0.25).
  std::vector<float> a = {std::log(0.9f), std::log(0.1f)};
  std::vector<float> b = {std::log(0.5f), std::log(0.5f)};
  for (auto [blend, expected] : {std::pair{"prob", 0.7}, {"log", 0.75}}) {
    WindowBlend w(1, 2, WindowBlend::Rows::kScores,
                  WindowConfig("center", blend));
    w.Add(a.data(), 1, 0);
    w.Add(b.data(), 1, 0);
    std::vector<float> out(2);
    ASSERT_TRUE(w.Finish(out.data()));
    EXPECT_NEAR(std::exp(out[0]), expected, 1e-5) << blend;
    EXPECT_NEAR(std::exp(out[1]), 1 - expected, 1e-5) << blend;
  }
}

TEST(WindowBlend, ReportsAnUncoveredFrame) {
  std::vector<float> a = {1, 1};
  WindowBlend blend(3, 1, WindowBlend::Rows::kHidden,
                    WindowConfig("center", "log"));
  blend.Add(a.data(), 2, 0);
  std::vector<float> out(3);
  EXPECT_FALSE(blend.Finish(out.data()));
}

TEST(WindowBlend, CenterPowerIncludesUniformAndSharpWeights) {
  const std::vector<float> a(6, 0), b(6, 1);
  for (float power : {0.0f, 0.5f, 1.0f, 2.0f, 1000.0f}) {
    auto config = WindowConfig("center", "log");
    config.center_power = power;
    WindowBlend blend(8, 1, WindowBlend::Rows::kHidden, config);
    blend.Add(a.data(), 6, 0);
    blend.Add(b.data(), 6, 2);
    std::vector<float> out(8);
    ASSERT_TRUE(blend.Finish(out.data()));
    const double expected = power > 100 ? 0 : 1 / (1 + std::pow(3, power));
    EXPECT_NEAR(out[2], expected, 1e-6) << power;
    EXPECT_NEAR(out[5], 1 - expected, 1e-6) << power;
    EXPECT_EQ(out.front(), 0);
    EXPECT_EQ(out.back(), 1);
  }
}

TEST(WindowBlend, ContextEdgesAreExcludedAndRecordingEndsRetained) {
  std::vector<float> a(6, 0), b(6, 1);
  a.back() = b.front() = std::numeric_limits<float>::quiet_NaN();
  for (bool fraction : {false, true}) {
    auto config = WindowConfig("center", "log");
    config.context_frames = fraction ? 0 : 1;
    config.context_fraction = fraction ? 0.2f : 0;
    config.center_power = 0;
    WindowBlend blend(8, 1, WindowBlend::Rows::kHidden, config);
    blend.Add(a.data(), 6, 0);
    blend.Add(b.data(), 6, 2);
    std::vector<float> out(8);
    ASSERT_TRUE(blend.Finish(out.data()));
    EXPECT_EQ(out, (std::vector<float>{0, 0, 0, 0.5, 0.5, 1, 1, 1}));
  }
}

TEST(WindowBlend, ContextCannotLeaveHolesOrDiscardASingleWindow) {
  auto config = WindowConfig("center", "log");
  config.context_frames = 3;
  std::vector<float> rows(6, 1), out(8, -9);
  WindowBlend blend(8, 1, WindowBlend::Rows::kHidden, config);
  blend.Add(rows.data(), 6, 0);
  blend.Add(rows.data(), 6, 2);
  EXPECT_FALSE(blend.Finish(out.data()));
  EXPECT_EQ(out, std::vector<float>(8, -9));
  WindowBlend single(6, 1, WindowBlend::Rows::kHidden, config);
  single.Add(rows.data(), 6, 0);
  ASSERT_TRUE(single.Finish(out.data()));
  EXPECT_TRUE(std::all_of(out.begin(), out.begin() + 6,
                          [](float x) { return x == 1; }));
}

TEST(WindowBlend, PoweredProbabilityAverageIsNormalizedAndOrderIndependent) {
  for (float power : {0.0f, 0.5f, 2.0f, 1000.0f}) {
    auto config = WindowConfig("center", "prob");
    config.center_power = power;
    config.context_frames = 1;
    std::vector<float> a, b;
    for (int i = 0; i < 6; ++i) {
      a.insert(a.end(), {std::log(0.9f), std::log(0.1f)});
      b.insert(b.end(), {std::log(0.5f), std::log(0.5f)});
    }
    WindowBlend forward(8, 2, WindowBlend::Rows::kScores, config);
    WindowBlend reverse(8, 2, WindowBlend::Rows::kScores, config);
    forward.Add(a.data(), 6, 0);
    forward.Add(b.data(), 6, 2);
    reverse.Add(b.data(), 6, 2);
    reverse.Add(a.data(), 6, 0);
    std::vector<float> x(16), y(16);
    ASSERT_TRUE(forward.Finish(x.data()));
    ASSERT_TRUE(reverse.Finish(y.data()));
    for (int i = 0; i < 16; ++i) EXPECT_NEAR(x[i], y[i], 1e-6);
    for (int i = 0; i < 8; ++i)
      EXPECT_NEAR(std::exp(x[2 * i]) + std::exp(x[2 * i + 1]), 1, 1e-6);
    const double ratio = power > 100 ? 0 : 1 / (1 + std::pow(2, power));
    EXPECT_NEAR(std::exp(x[6]), 0.9 * (1 - ratio) + 0.5 * ratio, 1e-6);
  }
}

TEST(OfflineEncoderWindowConfig, RejectsInvalidCenterControls) {
  OfflineEncoderWindowConfig config;
  EXPECT_TRUE(config.Validate());
  config.center_power = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(config.Validate());
  config.center_power = -1;
  EXPECT_FALSE(config.Validate());
  config.center_power = 0;
  EXPECT_TRUE(config.Validate());
  config.context_fraction = 0.5;
  EXPECT_FALSE(config.Validate());
  config.context_fraction = 0.1;
  config.context_frames = 1;
  EXPECT_FALSE(config.Validate());
  config.context_frames = 0;
  EXPECT_TRUE(config.Validate());
  config.weight = "cut";
  EXPECT_FALSE(config.Validate());
}

TEST(EvenWindowStarts, RetainsEdgeLossCoverageThroughSlackAndAlignment) {
  constexpr int32_t grid = 4;
  constexpr int32_t edge_loss = 16;
  for (int32_t window : {100, 101, 103}) {
    for (int32_t length = 2 * window; length < 4 * window; ++length) {
      WindowSchedule schedule{window, edge_loss, 8};
      std::vector<float> loudness(length, 1);
      loudness[92] = loudness[191] = 0;
      auto starts =
          EvenWindowStarts(length, window, schedule, loudness, grid, edge_loss);
      std::vector<int32_t> frames;
      int32_t total = 0;
      for (int32_t start : starts) {
        frames.push_back(
            (std::min(window, length - start) - edge_loss + grid - 1) / grid);
        total = std::max(total, start / grid + frames.back());
      }
      WindowBlend blend(total, 1, WindowBlend::Rows::kHidden,
                        OfflineEncoderWindowConfig{});
      for (size_t i = 0; i != starts.size(); ++i) {
        std::vector<float> rows(frames[i], 1);
        blend.Add(rows.data(), frames[i], starts[i] / grid);
      }
      std::vector<float> joined(total);
      ASSERT_TRUE(blend.Finish(joined.data())) << window << " " << length;
      EXPECT_EQ(joined, std::vector<float>(total, 1));
    }
  }
}

namespace {

constexpr int32_t kGrid = 4;
// Input frames lost at the end, as a convolutional front end loses them.
constexpr int32_t kEdge = 3;

// A stand-in encoder that is local in time, so that windows cannot change
// its output: output frame j of an item is its input frame kGrid * j + kEdge,
// for the item's valid length. Records the padded lengths it was given.
struct LocalEncoder {
  std::vector<int64_t> padded_lengths;
  std::vector<int64_t> batch_sizes;
  Ort::AllocatorWithDefaultOptions allocator;

  std::vector<Ort::Value> Encode(Ort::Value x, Ort::Value x_length) {
    auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t n = shape[0], t = shape[1], dim = shape[2];
    padded_lengths.push_back(t);
    batch_sizes.push_back(n);
    const int64_t *lengths = x_length.GetTensorData<int64_t>();
    int64_t max_out = 0;
    std::vector<int64_t> out_lengths(n);
    for (int64_t i = 0; i != n; ++i) {
      out_lengths[i] = std::max<int64_t>(0, (lengths[i] - kEdge) / kGrid);
      max_out = std::max(max_out, out_lengths[i]);
    }
    std::array<int64_t, 3> out_shape = {n, max_out, dim};
    Ort::Value out = Ort::Value::CreateTensor<float>(
        allocator, out_shape.data(), out_shape.size());
    float *p = out.GetTensorMutableData<float>();
    std::fill(p, p + n * max_out * dim, 0.0f);
    const float *in = x.GetTensorData<float>();
    for (int64_t i = 0; i != n; ++i) {
      for (int64_t j = 0; j != out_lengths[i]; ++j) {
        const float *from = in + (i * t + kGrid * j + kEdge) * dim;
        std::copy(from, from + dim, p + (i * max_out + j) * dim);
      }
    }
    std::array<int64_t, 1> length_shape = {n};
    Ort::Value out_length = Ort::Value::CreateTensor<int64_t>(
        allocator, length_shape.data(), length_shape.size());
    std::copy(out_lengths.begin(), out_lengths.end(),
              out_length.GetTensorMutableData<int64_t>());
    std::vector<Ort::Value> ans;
    ans.push_back(std::move(out));
    ans.push_back(std::move(out_length));
    return ans;
  }
};

std::unique_ptr<OfflineStream> MakeStream(const FeatureExtractorConfig &feat,
                                          float seconds, int32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> noise(0, 1);
  std::vector<float> samples(static_cast<size_t>(seconds * 16000));
  for (size_t i = 0; i != samples.size(); ++i) {
    // Loudness varying over time, so that quiet-edge placement has work.
    samples[i] = noise(rng) * 0.1f * (1.1f + std::sin(i / 4000.0f));
  }
  auto s = std::make_unique<OfflineStream>(feat);
  s->AcceptWaveform(16000, samples.data(), samples.size());
  return s;
}

// Each stream's encoder output, encoded with `config` and `normalize`, from
// streams of `feat` as a recognizer sets them up.
std::vector<std::vector<float>> EncodeAll(
    const OfflineEncoderConfig &config, bool batch,
    std::function<void(float *, int32_t, int32_t)> normalize,
    LocalEncoder *encoder, FeatureExtractorConfig feat = {}) {
  OfflineEncoderAdapter adapter;
  adapter.encode = [encoder](Ort::Value x, Ort::Value x_length) {
    return encoder->Encode(std::move(x), std::move(x_length));
  };
  adapter.normalize = std::move(normalize);
  adapter.subsampling_factor = kGrid;
  adapter.batch = batch;
  adapter.allocator = encoder->allocator;
  MoveStreamNormalizationToEncoder(config, &feat, &adapter);
  OfflineStreamEncoder e(std::move(adapter), config, feat.frame_shift_ms);
  EXPECT_EQ(e.Active(), config.window.Enabled() || !config.lengths.empty() ||
                            config.pad_to > 0);

  std::vector<std::unique_ptr<OfflineStream>> streams;
  streams.push_back(MakeStream(feat, 2.5, 1));
  streams.push_back(MakeStream(feat, 13.3, 2));
  streams.push_back(MakeStream(feat, 31.7, 3));
  std::vector<OfflineStream *> ss;
  for (auto &s : streams) ss.push_back(s.get());
  e.Encode(ss.data(), ss.size());

  std::vector<std::vector<float>> out;
  for (auto *s : ss) {
    EXPECT_TRUE(s->HasEncoderOutput());
    int32_t item = 0;
    auto batch = s->TakeEncoderOutput(&item);
    const float *row = batch->Row(item);
    out.emplace_back(
        row, row + static_cast<size_t>(batch->Frames(item)) * batch->Dim());
  }
  return out;
}

void ExpectNear(const std::vector<std::vector<float>> &a,
                const std::vector<std::vector<float>> &b, float tolerance) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i != a.size(); ++i) {
    ASSERT_EQ(a[i].size(), b[i].size()) << "stream " << i;
    for (size_t k = 0; k != a[i].size(); ++k) {
      ASSERT_NEAR(a[i][k], b[i][k], tolerance) << "stream " << i << " " << k;
    }
  }
}

// Subtracts each dimension's mean over the given frames: a normalization
// that uses its input's statistics.
void SubtractMean(float *f, int32_t num_frames, int32_t dim) {
  for (int32_t d = 0; d != dim; ++d) {
    double sum = 0;
    for (int32_t t = 0; t != num_frames; ++t) sum += f[t * dim + d];
    for (int32_t t = 0; t != num_frames; ++t) {
      f[t * dim + d] -= static_cast<float>(sum / num_frames);
    }
  }
}

OfflineEncoderConfig Windowed(const char *weight, const char *normalize) {
  OfflineEncoderConfig c;
  c.window.seconds = 10;
  c.window.weight = weight;
  c.window.normalize = normalize;
  return c;
}

}  // namespace

TEST(OfflineStreamEncoder, WindowsGiveTheWholeUtterancesOutputForALocalModel) {
  LocalEncoder whole_encoder;
  auto whole = EncodeAll({}, true, nullptr, &whole_encoder);
  EXPECT_EQ(whole_encoder.batch_sizes, std::vector<int64_t>{3});

  for (const char *weight : {"center", "cut"}) {
    for (bool batch : {true, false}) {
      LocalEncoder encoder;
      auto windowed =
          EncodeAll(Windowed(weight, "window"), batch, nullptr, &encoder);
      ExpectNear(windowed, whole, 1e-5);
      // The 2.5 s stream whole; 2 windows of 10 s for 13.3 s, 5 for 31.7 s.
      int64_t items = 0;
      for (auto n : encoder.batch_sizes) items += n;
      EXPECT_EQ(items, 8) << weight << " " << batch;
      EXPECT_EQ(encoder.batch_sizes.size(), batch ? 1u : 8u);
    }
  }
}

TEST(OfflineStreamEncoder, NormalizationScopeIsAnOption) {
  LocalEncoder e1, e2, e3;
  auto whole = EncodeAll({}, true, SubtractMean, &e1);
  auto per_utterance =
      EncodeAll(Windowed("center", "utterance"), true, SubtractMean, &e2);
  auto per_window =
      EncodeAll(Windowed("center", "window"), true, SubtractMean, &e3);
  ExpectNear(per_utterance, whole, 1e-4);
  // The short stream is one window either way; the long ones differ.
  ExpectNear({per_window[0]}, {whole[0]}, 1e-6);
  float largest = 0;
  for (size_t k = 0; k != whole[2].size(); ++k) {
    largest = std::max(largest, std::abs(per_window[2][k] - whole[2][k]));
  }
  EXPECT_GT(largest, 1e-3);
}

TEST(OfflineStreamEncoder, NemoNormalizationMovesFromTheStreamToTheWindows) {
  FeatureExtractorConfig nemo;
  nemo.nemo_normalize_type = "per_feature";
  LocalEncoder e1, e2, e3;
  auto whole = EncodeAll({}, true, nullptr, &e1, nemo);
  // Over the utterance, in the encoder instead of the stream: the same.
  auto per_utterance =
      EncodeAll(Windowed("center", "utterance"), true, nullptr, &e2, nemo);
  ExpectNear(per_utterance, whole, 1e-4);
  // Per window: a short utterance is its own window; long ones differ.
  auto per_window =
      EncodeAll(Windowed("center", "window"), true, nullptr, &e3, nemo);
  ExpectNear({per_window[0]}, {whole[0]}, 1e-5);
  float largest = 0;
  for (size_t k = 0; k != whole[2].size(); ++k) {
    largest = std::max(largest, std::abs(per_window[2][k] - whole[2][k]));
  }
  EXPECT_GT(largest, 1e-3);
}

TEST(OfflineStreamEncoder, PadsInputsToTheAllowedLengths) {
  LocalEncoder plain, listed;
  auto unpadded =
      EncodeAll(Windowed("center", "window"), true, nullptr, &plain);
  OfflineEncoderConfig c = Windowed("center", "window");
  c.lengths = "5,12";
  c.window.seconds = 11;  // the window becomes the 12 s length
  auto padded = EncodeAll(c, true, nullptr, &listed);
  EXPECT_EQ(listed.padded_lengths, std::vector<int64_t>{1200});
  // Different windows, same local model: the same output.
  ExpectNear(padded, unpadded, 1e-5);

  LocalEncoder rounded;
  OfflineEncoderConfig whole;
  whole.pad_to = 4;
  auto out = EncodeAll(whole, true, nullptr, &rounded);
  EXPECT_EQ(rounded.padded_lengths, std::vector<int64_t>{3200});
  LocalEncoder plain_whole;
  ExpectNear(out, EncodeAll({}, true, nullptr, &plain_whole), 0);
}

// A batch of three items, (3, 4, 2), whose item i has i + 2 valid frames,
// value 10 * item + frame, and nonzero padding.
std::shared_ptr<const EncoderOutputBatch> ThreeItems(OrtAllocator *allocator) {
  std::array<int64_t, 3> shape = {3, 4, 2};
  Ort::Value values =
      Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
  float *p = values.GetTensorMutableData<float>();
  std::fill(p, p + 24, -7.0f);
  std::array<int64_t, 1> length_shape = {3};
  Ort::Value lengths = Ort::Value::CreateTensor<int64_t>(
      allocator, length_shape.data(), length_shape.size());
  for (int64_t i = 0; i != 3; ++i) {
    lengths.GetTensorMutableData<int64_t>()[i] = i + 2;
    for (int64_t t = 0; t != i + 2; ++t) {
      p[(i * 4 + t) * 2] = p[(i * 4 + t) * 2 + 1] = 10.0f * i + t;
    }
  }
  return std::make_shared<const EncoderOutputBatch>(std::move(values), lengths);
}

TEST(TakeEncoderOutputs, ViewsConsecutiveItemsAndCopiesOtherwise) {
  Ort::AllocatorWithDefaultOptions allocator;
  FeatureExtractorConfig feat;
  auto batch = ThreeItems(allocator);
  std::vector<std::unique_ptr<OfflineStream>> streams;
  for (int i = 0; i != 3; ++i) {
    streams.push_back(std::make_unique<OfflineStream>(feat));
  }

  // Items 1 and 2, in order: the batch's own rows.
  for (int i = 1; i != 3; ++i) streams[i]->SetEncoderOutput(batch, i);
  OfflineStream *tail[] = {streams[1].get(), streams[2].get()};
  auto view = TakeEncoderOutputs(tail, 2, allocator);
  EXPECT_EQ(view.values.GetTensorData<float>(), batch->Row(1));
  EXPECT_EQ(view.values.GetTensorTypeAndShapeInfo().GetShape(),
            (std::vector<int64_t>{2, 4, 2}));
  EXPECT_EQ(view.lengths.GetTensorData<int64_t>()[1], 4);
  EXPECT_FALSE(streams[1]->HasEncoderOutput());
  ASSERT_EQ(view.batches.size(), 1u);

  // Items 2 and 0: a padded copy, padding zeroed.
  streams[0]->SetEncoderOutput(batch, 2);
  streams[1]->SetEncoderOutput(batch, 0);
  OfflineStream *swapped[] = {streams[0].get(), streams[1].get()};
  auto copy = TakeEncoderOutputs(swapped, 2, allocator);
  EXPECT_NE(copy.values.GetTensorData<float>(), batch->Row(2));
  EXPECT_EQ(copy.values.GetTensorTypeAndShapeInfo().GetShape(),
            (std::vector<int64_t>{2, 4, 2}));
  const float *c = copy.values.GetTensorData<float>();
  EXPECT_EQ(c[3 * 2], 23.0f);       // item 2, frame 3
  EXPECT_EQ(c[(4 + 1) * 2], 1.0f);  // item 0, frame 1
  EXPECT_EQ(c[(4 + 2) * 2], 0.0f);  // item 0's padding
  EXPECT_TRUE(copy.batches.empty());
}

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/nemo-transducer-prefix-segment-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/nemo-transducer-prefix-segment.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kVocab = 5;  // Four tokens and blank, the last.
constexpr int32_t kEncoderDim = 3;
constexpr int32_t kHidden = 2;
const char kTokens[] = "a 0\nb 1\nc 2\nd 3\n<blk> 4\n";

// A stand-in for a NeMo RNN-T: the prediction network's state is one
// number per hypothesis, updated from the token; the joiner's logits
// depend on the frame and on that prediction. "one_at_a_time" makes both
// refuse several hypotheses per call, as some exported models do.
struct FakeModel {
  bool one_at_a_time = false;
  std::atomic<int32_t> decoder_calls{0};
  std::atomic<int32_t> joiner_calls{0};
  Ort::AllocatorWithDefaultOptions allocator;

  void Refuse(int64_t num) const {
    if (one_at_a_time && num > 1) {
      throw Ort::Exception("one hypothesis per call", ORT_INVALID_ARGUMENT);
    }
  }

  std::pair<Ort::Value, std::vector<Ort::Value>> RunDecoder(
      Ort::Value tokens, std::vector<Ort::Value> states) {
    int64_t num = tokens.GetTensorTypeAndShapeInfo().GetShape()[0];
    Refuse(num);
    ++decoder_calls;
    std::array<int64_t, 3> out_shape = {num, kHidden, 1};
    Ort::Value out = Ort::Value::CreateTensor<float>(
        allocator, out_shape.data(), out_shape.size());
    std::array<int64_t, 3> state_shape = {1, num, 1};
    Ort::Value next = Ort::Value::CreateTensor<float>(
        allocator, state_shape.data(), state_shape.size());
    for (int64_t i = 0; i != num; ++i) {
      int32_t token = tokens.GetTensorData<int32_t>()[i];
      float state = 0.6f * states[0].GetTensorData<float>()[i] +
                    0.37f * static_cast<float>(token + 1);
      next.GetTensorMutableData<float>()[i] = state;
      out.GetTensorMutableData<float>()[i * kHidden] = std::sin(state);
      out.GetTensorMutableData<float>()[i * kHidden + 1] = std::cos(2 * state);
    }
    std::vector<Ort::Value> next_states;
    next_states.push_back(std::move(next));
    return {std::move(out), std::move(next_states)};
  }

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) {
    int64_t num = encoder_out.GetTensorTypeAndShapeInfo().GetShape()[0];
    Refuse(num);
    ++joiner_calls;
    std::array<int64_t, 4> shape = {num, 1, 1, kVocab};
    Ort::Value logits =
        Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
    for (int64_t i = 0; i != num; ++i) {
      const float *e = encoder_out.GetTensorData<float>() + i * kEncoderDim;
      const float *d = decoder_out.GetTensorData<float>() + i * kHidden;
      for (int32_t v = 0; v != kVocab; ++v) {
        logits.GetTensorMutableData<float>()[i * kVocab + v] =
            6.0f * std::sin(e[0] * (v + 1) + e[1]) + d[0] * (v - 2) +
            d[1] * e[2] + (v == kVocab - 1 ? 2.0f : 0.0f);
      }
    }
    return logits;
  }

  std::vector<Ort::Value> InitStates() {
    std::array<int64_t, 3> shape = {1, 1, 1};
    std::vector<Ort::Value> states;
    states.push_back(
        Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size()));
    states[0].GetTensorMutableData<float>()[0] = 0;
    return states;
  }

  void Configure(NeMoTransducerCalls *calls) {
    calls->run_decoder = [this](Ort::Value tokens,
                                std::vector<Ort::Value> states) {
      return RunDecoder(std::move(tokens), std::move(states));
    };
    calls->run_joiner = [this](Ort::Value encoder_out, Ort::Value decoder_out) {
      return RunJoiner(std::move(encoder_out), std::move(decoder_out));
    };
    calls->init_states = [this]() { return InitStates(); };
    calls->allocator = allocator;
    calls->vocab_size = kVocab;
  }
};

std::vector<float> Frames(int32_t num_frames) {
  std::vector<float> frames(num_frames * kEncoderDim);
  for (size_t i = 0; i != frames.size(); ++i) {
    frames[i] = std::sin(0.7f * static_cast<float>(i) + 0.2f);
  }
  return frames;
}

struct Decoded {
  std::vector<int32_t> tokens;
  float score = 0;
  int32_t decoder_calls = 0;
  int32_t joiner_calls = 0;
};

// Decodes num_frames frames in chunks of chunk_frames.
Decoded Decode(bool one_at_a_time, int32_t num_frames, int32_t chunk_frames) {
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  TransducerFusion fusion(config, symbol_table, kVocab - 1, kVocab);
  FakeModel model;
  model.one_at_a_time = one_at_a_time;
  NeMoTransducerCalls calls;
  model.Configure(&calls);
  NeMoTransducerPrefixSegment segment(&calls, &fusion, config, 4, 0.0f);
  auto frames = Frames(num_frames);
  for (int32_t t = 0; t < num_frames; t += chunk_frames) {
    segment.Decode(frames.data() + t * kEncoderDim,
                   std::min(chunk_frames, num_frames - t), kEncoderDim);
  }
  auto best = segment.Search()->Finish(1);
  Decoded decoded;
  if (!best.empty()) {
    decoded.tokens = best[0].tokens;
    decoded.score = best[0].score;
  }
  decoded.decoder_calls = model.decoder_calls;
  decoded.joiner_calls = model.joiner_calls;
  EXPECT_EQ(calls.joiner_batches.load(), !one_at_a_time);
  return decoded;
}

TEST(NeMoTransducerPrefixSegment, ChunksGiveTheWholeUtterancesResult) {
  const int32_t num_frames = 24;
  Decoded whole = Decode(false, num_frames, num_frames);
  ASSERT_FALSE(whole.tokens.empty());
  for (int32_t chunk : {1, 5, 7}) {
    Decoded chunked = Decode(false, num_frames, chunk);
    EXPECT_EQ(chunked.tokens, whole.tokens) << "chunk " << chunk;
    EXPECT_NEAR(chunked.score, whole.score, 1e-4);
  }
}

// A model that takes one hypothesis per call gives the same result, with
// more calls.
TEST(NeMoTransducerPrefixSegment, OneHypothesisPerCallGivesTheSameResult) {
  const int32_t num_frames = 24;
  Decoded batched = Decode(false, num_frames, num_frames);
  Decoded single = Decode(true, num_frames, num_frames);
  EXPECT_EQ(single.tokens, batched.tokens);
  EXPECT_NEAR(single.score, batched.score, 1e-4);
  EXPECT_GT(single.joiner_calls, batched.joiner_calls);
  EXPECT_GE(single.decoder_calls, batched.decoder_calls);
}

TEST(NeMoTransducerPrefixSegment, ThreadsShareFixedBatchFallback) {
  const int32_t num_frames = 24;
  Decoded expected = Decode(true, num_frames, num_frames);
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  TransducerFusion fusion(config, symbol_table, kVocab - 1, kVocab);
  FakeModel model;
  model.one_at_a_time = true;
  NeMoTransducerCalls calls;
  model.Configure(&calls);
  auto frames = Frames(num_frames);
  std::array<std::vector<TransducerPrefixHypothesis>, 4> results;
  std::vector<std::thread> threads;
  for (size_t i = 0; i != results.size(); ++i) {
    threads.emplace_back([&, i]() {
      NeMoTransducerPrefixSegment segment(&calls, &fusion, config, 4, 0.0f);
      segment.Decode(frames.data(), num_frames, kEncoderDim);
      results[i] = segment.Search()->Finish(1);
    });
  }
  for (auto &thread : threads) thread.join();
  for (const auto &result : results) {
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].tokens, expected.tokens);
    EXPECT_NEAR(result[0].score, expected.score, 1e-4);
  }
  EXPECT_FALSE(calls.decoder_batches.load());
  EXPECT_FALSE(calls.joiner_batches.load());
}

}  // namespace
}  // namespace sherpa_onnx

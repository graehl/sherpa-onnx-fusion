// sherpa-onnx/csrc/ctc-inflight-decoder-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/ctc-inflight-decoder.h"

#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kVocabSize = 4;

// A HistoryScorer that records how its work was batched.
class CountingScorer : public HistoryScorer {
 public:
  CountingScorer() : HistoryScorer(ScorerUnit::kCharacter, -0.1f) {}

  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) override {
    calls += 1;
    items += static_cast<int32_t>(extensions.size);
    HistoryScorer::Advance(extensions, results);
  }

  int32_t calls = 0;
  int32_t items = 0;
};

std::shared_ptr<CtcSearchVocabulary> Vocabulary() {
  auto vocabulary = std::make_shared<CtcSearchVocabulary>();
  vocabulary->tokens.resize(kVocabSize);
  vocabulary->tokens[1].starts_word = true;
  vocabulary->tokens[1].chars = {'a', 'b'};
  vocabulary->tokens[2].chars = {'c'};
  vocabulary->tokens[3].starts_word = true;
  return vocabulary;
}

std::vector<float> RandomLogProbs(int32_t num_frames, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-3.0f, 3.0f);
  std::vector<float> log_probs(num_frames * kVocabSize);
  for (int32_t t = 0; t != num_frames; ++t) {
    float *row = log_probs.data() + t * kVocabSize;
    float total = 0;
    for (int32_t v = 0; v != kVocabSize; ++v) {
      row[v] = uniform(rng);
      total += std::exp(row[v]);
    }
    for (int32_t v = 0; v != kVocabSize; ++v) row[v] -= std::log(total);
  }
  return log_probs;
}

CtcInflightDecoderOptions Options() {
  CtcInflightDecoderOptions options;
  options.search.max_active = 4;
  options.scorer_weights = {0.5f};
  return options;
}

// What decoding "log_probs" alone gives.
std::vector<CtcPrefixHypothesis> Alone(const std::vector<float> &log_probs,
                                       float weight, int32_t nbest) {
  HistoryScorer scorer(ScorerUnit::kCharacter, -0.1f);
  CtcPrefixSearch search(Options().search, Vocabulary(), {&scorer}, {weight});
  search.Decode(log_probs.data(), log_probs.size() / kVocabSize);
  return search.Finish(nbest);
}

void ExpectSame(const std::vector<CtcPrefixHypothesis> &a,
                const std::vector<CtcPrefixHypothesis> &b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a[i].tokens, b[i].tokens);
    EXPECT_FLOAT_EQ(a[i].score, b[i].score);
  }
}

TEST(CtcInflightDecoder, StaggeredUtterancesMatchDecodingEachAlone) {
  auto scorer = std::make_shared<CountingScorer>();
  auto a_frames = RandomLogProbs(10, 1);
  auto b_frames = RandomLogProbs(4, 2);
  auto c_frames = RandomLogProbs(7, 3);
  {
    CtcInflightDecoder decoder(Options(), Vocabulary(), {scorer});

    // A starts alone and gets its input in two chunks.
    UtteranceHandle a = *decoder.TryAdd();
    ASSERT_TRUE(decoder.TryAccept(a, a_frames.data(), 3));
    StepProgress progress = decoder.Step();
    EXPECT_EQ(progress.frames_decoded, 1);
    EXPECT_TRUE(progress.has_ready_work);

    // B joins with all its input and its own weight; C joins and waits.
    UtteranceOptions b_options;
    b_options.scorer_weights = {0.9f};
    b_options.nbest = 3;
    UtteranceHandle b = *decoder.TryAdd(b_options);
    ASSERT_TRUE(decoder.TryAccept(b, b_frames.data(), 4));
    decoder.InputFinished(b);
    UtteranceHandle c = *decoder.TryAdd();
    EXPECT_EQ(decoder.NumActiveUtterances(), 3);

    // A runs out of queued frames after two more steps; B continues.
    decoder.Step();
    progress = decoder.Step();
    EXPECT_EQ(progress.frames_decoded, 2);
    EXPECT_TRUE(progress.waiting_for_input);
    progress = decoder.Step();
    EXPECT_EQ(progress.frames_decoded, 1);
    EXPECT_EQ(decoder.Status(a), DecodeStatus::kActive);
    EXPECT_EQ(decoder.Status(b), DecodeStatus::kActive);

    // B's last frame is decoded and B completes in the same step, while A
    // and C are still active.
    progress = decoder.Step();
    EXPECT_EQ(progress.frames_decoded, 1);
    EXPECT_EQ(decoder.Status(b), DecodeStatus::kComplete);
    EXPECT_EQ(decoder.NumActiveUtterances(), 2);
    ExpectSame(decoder.GetResult(b), Alone(b_frames, 0.9f, 3));

    // The rest of A and all of C arrive.
    ASSERT_TRUE(decoder.TryAccept(a, a_frames.data() + 3 * kVocabSize, 7));
    decoder.InputFinished(a);
    ASSERT_TRUE(decoder.TryAccept(c, c_frames.data(), 7));
    decoder.InputFinished(c);
    EXPECT_EQ(decoder.GetResult(a).size(), 1u);  // Partial result.
    while (decoder.NumActiveUtterances() != 0) decoder.Step();

    EXPECT_EQ(decoder.Status(a), DecodeStatus::kComplete);
    ExpectSame(decoder.GetResult(a), Alone(a_frames, 0.5f, 1));
    ExpectSame(decoder.GetResult(c), Alone(c_frames, 0.5f, 1));

    // Scorer calls served several utterances at once.
    EXPECT_GT(scorer->items, scorer->calls);

    decoder.Release(a);
    decoder.Release(b);
    EXPECT_THROW(decoder.Status(a), std::out_of_range);
    EXPECT_EQ(decoder.Status(c), DecodeStatus::kComplete);
  }
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

TEST(CtcInflightDecoder, CancelLeavesOtherUtterancesRunning) {
  auto scorer = std::make_shared<CountingScorer>();
  auto a_frames = RandomLogProbs(6, 11);
  auto b_frames = RandomLogProbs(6, 12);
  CtcInflightDecoder decoder(Options(), Vocabulary(), {scorer});
  UtteranceHandle a = *decoder.TryAdd();
  UtteranceHandle b = *decoder.TryAdd();
  ASSERT_TRUE(decoder.TryAccept(a, a_frames.data(), 6));
  ASSERT_TRUE(decoder.TryAccept(b, b_frames.data(), 6));
  decoder.InputFinished(a);
  decoder.InputFinished(b);
  decoder.Step();
  decoder.Step();

  EXPECT_THROW(decoder.Release(b), std::invalid_argument);
  decoder.Cancel(b);
  EXPECT_EQ(decoder.Status(b), DecodeStatus::kCancelled);
  EXPECT_TRUE(decoder.GetResult(b).empty());
  EXPECT_FALSE(decoder.TryAccept(b, b_frames.data(), 1));
  decoder.Release(b);

  while (decoder.NumActiveUtterances() != 0) decoder.Step();
  ExpectSame(decoder.GetResult(a), Alone(a_frames, 0.5f, 1));
}

TEST(CtcInflightDecoder, RefusesWorkBeyondItsCapacity) {
  CtcInflightDecoderOptions options = Options();
  options.max_active_utterances = 2;
  options.max_queued_input_bytes = 5 * kVocabSize * sizeof(float);
  auto frames = RandomLogProbs(8, 21);
  CtcInflightDecoder decoder(options, Vocabulary(),
                             {std::make_shared<CountingScorer>()});

  UtteranceHandle a = *decoder.TryAdd();
  UtteranceHandle b = *decoder.TryAdd();
  EXPECT_FALSE(decoder.TryAdd().has_value());

  // Four frames fit; two more do not until some are decoded.
  ASSERT_TRUE(decoder.TryAccept(a, frames.data(), 4));
  EXPECT_FALSE(decoder.TryAccept(b, frames.data(), 2));
  decoder.Step();
  EXPECT_TRUE(decoder.TryAccept(b, frames.data(), 2));
  // A chunk that could never fit is an error, not backpressure.
  EXPECT_THROW(decoder.TryAccept(b, frames.data(), 6), std::invalid_argument);

  // Finishing an utterance frees a place.
  decoder.Cancel(a);
  EXPECT_TRUE(decoder.TryAdd().has_value());

  UtteranceOptions wrong;
  wrong.scorer_weights = {0.1f, 0.2f};
  EXPECT_THROW(decoder.TryAdd(wrong), std::invalid_argument);
}

TEST(CtcInflightDecoder, ReleasedHandlesStayInvalidAfterReadmission) {
  auto options = Options();
  options.max_active_utterances = 1;
  CtcInflightDecoder decoder(options, Vocabulary(),
                             {std::make_shared<CountingScorer>()});
  const auto first = *decoder.TryAdd();
  decoder.Cancel(first);
  decoder.Release(first);
  for (int32_t i = 0; i != 1000; ++i) {
    const auto handle = *decoder.TryAdd();
    EXPECT_GT(handle.id, first.id);
    EXPECT_THROW(decoder.Status(first), std::out_of_range);
    decoder.InputFinished(handle);
    decoder.Step();
    EXPECT_EQ(decoder.Status(handle), DecodeStatus::kComplete);
    decoder.Release(handle);
  }
  EXPECT_EQ(decoder.NumActiveUtterances(), 0);
  EXPECT_FALSE(decoder.Step().has_ready_work);
}

TEST(CtcInflightDecoder, RejectedPrefixesLeaveTheEmptyOutput) {
  auto scorer = std::make_shared<BigramTableScorer>(ScorerUnit::kAsrToken);
  for (int32_t token = 1; token != kVocabSize; ++token) {
    scorer->Reject(BigramTableScorer::kStart, token);
  }
  // The scorer rejects every first token, so the empty output is the only
  // hypothesis it accepts.
  std::vector<float> frames(3 * kVocabSize, std::log(0.33f));
  CtcInflightDecoderOptions options = Options();
  CtcInflightDecoder decoder(options, Vocabulary(), {scorer});
  UtteranceHandle a = *decoder.TryAdd();
  ASSERT_TRUE(decoder.TryAccept(a, frames.data(), 3));
  decoder.InputFinished(a);
  while (decoder.NumActiveUtterances() != 0) decoder.Step();
  ASSERT_EQ(decoder.Status(a), DecodeStatus::kComplete);
  EXPECT_TRUE(decoder.GetResult(a)[0].tokens.empty());
}

}  // namespace
}  // namespace sherpa_onnx

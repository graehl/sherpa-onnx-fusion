// sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.h"

#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

// The token table: blank, "▁the", "re", "▁" alone, <unk>, "é".
const char kTokens[] = "<blk> 0\n▁the 1\nre 2\n▁ 3\n<unk> 4\né 5\n";
constexpr int32_t kVocabSize = 6;

// [batch, num_frames, kVocabSize] log probabilities.
std::vector<float> RandomLogProbs(int32_t batch, int32_t num_frames,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-3.0f, 3.0f);
  std::vector<float> log_probs(batch * num_frames * kVocabSize);
  for (size_t row = 0; row != log_probs.size() / kVocabSize; ++row) {
    float *p = log_probs.data() + row * kVocabSize;
    float total = 0;
    for (int32_t v = 0; v != kVocabSize; ++v) {
      p[v] = uniform(rng);
      total += std::exp(p[v]);
    }
    for (int32_t v = 0; v != kVocabSize; ++v) p[v] -= std::log(total);
  }
  return log_probs;
}

TEST(OnlineCtcPrefixBeamSearchDecoder, ChunksGiveTheSearchesBestPrefix) {
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  const int32_t max_active = 4, num_chunks = 3, chunk_frames = 4;

  OnlineCtcPrefixBeamSearchDecoder decoder(config, max_active, symbol_table, 0);
  OnlineStream first, second;
  OnlineStream *streams[] = {&first, &second};
  std::vector<OnlineCtcDecoderResult> results(2);

  auto vocabulary = MakeCtcSearchVocabulary(symbol_table, 0);
  CtcPrefixSearchOptions options =
      MakeCtcPrefixSearchOptions(config, max_active);
  CtcPrefixSearch expected_first(options, vocabulary, {}, {});
  CtcPrefixSearch expected_second(options, vocabulary, {}, {});
  CtcPrefixSearch *expected[] = {&expected_first, &expected_second};

  for (int32_t chunk = 0; chunk != num_chunks; ++chunk) {
    auto log_probs = RandomLogProbs(2, chunk_frames, 40 + chunk);
    decoder.Decode(log_probs.data(), 2, chunk_frames, kVocabSize, &results,
                   streams, 2);
    for (int32_t b = 0; b != 2; ++b) {
      expected[b]->Decode(log_probs.data() + b * chunk_frames * kVocabSize,
                          chunk_frames);
      CtcPrefixHypothesis best = expected[b]->Best();
      std::vector<int64_t> tokens(best.tokens.begin(), best.tokens.end());
      EXPECT_EQ(results[b].tokens, tokens) << "chunk " << chunk;
      EXPECT_EQ(results[b].timestamps, best.timestamps) << "chunk " << chunk;
    }
  }
  ASSERT_NE(first.GetCtcPrefixSearch(), nullptr);
  EXPECT_EQ(first.GetCtcPrefixSearch()->NumFramesDecoded(),
            num_chunks * chunk_frames);
}

TEST(OnlineCtcPrefixBeamSearchDecoder, CountsFramesSinceTheLastToken) {
  SymbolTable symbol_table(kTokens, false);
  OnlineCtcPrefixBeamSearchDecoder decoder(FusionConfig(), 4, symbol_table, 0);
  OnlineStream stream;
  OnlineStream *streams[] = {&stream};
  std::vector<OnlineCtcDecoderResult> results(1);

  auto frame = [](int32_t token) {
    std::vector<float> row(kVocabSize, -20.0f);
    row[token] = -1e-4f;
    return row;
  };
  auto decode = [&](const std::vector<int32_t> &tokens) {
    std::vector<float> log_probs;
    for (int32_t token : tokens) {
      auto row = frame(token);
      log_probs.insert(log_probs.end(), row.begin(), row.end());
    }
    decoder.Decode(log_probs.data(), 1, tokens.size(), kVocabSize, &results,
                   streams, 1);
  };

  decode({0, 0});
  EXPECT_TRUE(results[0].tokens.empty());
  EXPECT_EQ(results[0].num_trailing_blanks, 2);

  decode({1, 0, 0});
  EXPECT_EQ(results[0].tokens, std::vector<int64_t>({1}));
  EXPECT_EQ(results[0].timestamps, std::vector<int32_t>({2}));
  EXPECT_EQ(results[0].num_trailing_blanks, 2);

  decode({0, 0});
  EXPECT_EQ(results[0].num_trailing_blanks, 4);

  // The recognizer drops the stream's search at an endpoint; the next chunk
  // starts a new one.
  stream.SetCtcPrefixSearch(nullptr);
  decode({0, 2});
  EXPECT_EQ(results[0].tokens, std::vector<int64_t>({2}));
  EXPECT_EQ(results[0].timestamps, std::vector<int32_t>({1}));
  EXPECT_EQ(results[0].num_trailing_blanks, 0);
}

// A token bigram scorer that makes ending in token 1 costly.
class FinalPenaltyFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions & /*common*/,
                                 const std::string & /*extra*/) const override {
    auto scorer =
        std::make_unique<BigramTableScorer>(ScorerUnit::kAsrToken, 0.0f);
    scorer->SetFinal(1, -10.0f);
    return scorer;
  }
};

TEST(OnlineCtcPrefixBeamSearchDecoder, EndOfInputRevisesTheResult) {
  RegisterScorerFactory("test-final-penalty",
                        std::make_unique<FinalPenaltyFactory>());
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  config.scorers.resize(1);
  config.scorers[0].type = "test-final-penalty";
  OnlineCtcPrefixBeamSearchDecoder decoder(config, 4, symbol_table, 0);

  OnlineStream stream;
  OnlineStream *streams[] = {&stream};
  std::vector<OnlineCtcDecoderResult> results(1);
  std::vector<float> samples(16000, 0.0f);
  stream.AcceptWaveform(16000, samples.data(), samples.size());

  // One frame in which token 1 is a little more probable than token 2.
  std::vector<float> frame(kVocabSize, -20.0f);
  frame[1] = std::log(0.55f);
  frame[2] = std::log(0.45f);
  decoder.Decode(frame.data(), 1, 1, kVocabSize, &results, streams, 1);
  EXPECT_EQ(results[0].tokens, std::vector<int64_t>({1}));

  // A blank frame after the input ended: the end-of-input score now counts.
  stream.InputFinished();
  std::vector<float> blank(kVocabSize, -20.0f);
  blank[0] = -1e-4f;
  decoder.Decode(blank.data(), 1, 1, kVocabSize, &results, streams, 1);
  EXPECT_EQ(results[0].tokens, std::vector<int64_t>({2}));
}

class RejectFinalScorer : public BigramTableScorer {
 public:
  RejectFinalScorer() : BigramTableScorer(ScorerUnit::kAsrToken, 0.0f) {}

  void Finish(Span<const ScorerStateRef> states,
              ScorerFinalScore *results) override {
    for (size_t i = 0; i != states.size; ++i) {
      results[i] = {0.0f, false};
    }
  }
};

class RejectFinalFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions & /*common*/,
                                 const std::string & /*extra*/) const override {
    return std::make_unique<RejectFinalScorer>();
  }
};

TEST(OnlineCtcPrefixBeamSearchDecoder, RejectedFinalClearsPartialOutput) {
  RegisterScorerFactory("test-reject-final",
                        std::make_unique<RejectFinalFactory>());
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  config.scorers.resize(1);
  config.scorers[0].type = "test-reject-final";
  config.scorers[0].weight = 0.0f;
  OnlineCtcPrefixBeamSearchDecoder decoder(config, 4, symbol_table, 0);
  OnlineStream stream;
  OnlineStream *streams[] = {&stream};
  std::vector<OnlineCtcDecoderResult> results(1);
  std::vector<float> samples(16000, 0.0f);
  stream.AcceptWaveform(16000, samples.data(), samples.size());
  std::vector<float> frame(kVocabSize, -20.0f);
  frame[1] = 0.0f;
  decoder.Decode(frame.data(), 1, 1, kVocabSize, &results, streams, 1);
  ASSERT_EQ(results[0].tokens, std::vector<int64_t>({1}));

  stream.InputFinished();
  decoder.Decode(frame.data(), 1, 1, kVocabSize, &results, streams, 1);
  EXPECT_TRUE(results[0].tokens.empty());
  EXPECT_TRUE(results[0].timestamps.empty());
  EXPECT_EQ(results[0].num_trailing_blanks, 2);
}

int32_t live_counted_scorers = 0;

class CountedScorer : public BigramTableScorer {
 public:
  CountedScorer() : BigramTableScorer(ScorerUnit::kAsrToken, 0.0f) {
    ++live_counted_scorers;
  }
  ~CountedScorer() override { --live_counted_scorers; }
};

class CountedScorerFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions & /*common*/,
                                 const std::string & /*extra*/) const override {
    return std::make_unique<CountedScorer>();
  }
};

// A binding may destroy a recognizer before its streams; a stream's search
// still frees its scorer states through the scorers.
TEST(OnlineCtcPrefixBeamSearchDecoder, StreamMayOutliveTheDecoder) {
  RegisterScorerFactory("test-counted",
                        std::make_unique<CountedScorerFactory>());
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  config.scorers.resize(1);
  config.scorers[0].type = "test-counted";
  auto stream = std::make_unique<OnlineStream>();
  OnlineStream *streams[] = {stream.get()};
  std::vector<float> samples(16000, 0.0f);
  stream->AcceptWaveform(16000, samples.data(), samples.size());
  {
    OnlineCtcPrefixBeamSearchDecoder decoder(config, 4, symbol_table, 0);
    std::vector<OnlineCtcDecoderResult> results(1);
    std::vector<float> frame(kVocabSize, -20.0f);
    frame[1] = 0.0f;
    decoder.Decode(frame.data(), 1, 1, kVocabSize, &results, streams, 1);
  }
  EXPECT_EQ(live_counted_scorers, 1);
  stream.reset();
  EXPECT_EQ(live_counted_scorers, 0);
}

TEST(OnlineCtcPrefixBeamSearchDecoder, GraphAndPrefixSearchAreExclusive) {
  OfflineRecognizerConfig offline;
  offline.decoding_method = "prefix_beam_search";
  offline.model_config.tokens = __FILE__;
  offline.model_config.nemo_ctc.model = __FILE__;
  ASSERT_TRUE(offline.Validate());
  offline.ctc_fst_decoder_config.graph = __FILE__;
  EXPECT_FALSE(offline.Validate());

  OnlineRecognizerConfig online;
  online.decoding_method = "prefix_beam_search";
  online.model_config.tokens = __FILE__;
  online.model_config.nemo_ctc.model = __FILE__;
  ASSERT_TRUE(online.Validate());
  online.ctc_fst_decoder_config.graph = __FILE__;
  EXPECT_FALSE(online.Validate());
}

}  // namespace
}  // namespace sherpa_onnx

// sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

// The token table: blank, "▁the", "re", "▁" alone, <unk>, "é".
const char kTokens[] = "<blk> 0\n▁the 1\nre 2\n▁ 3\n<unk> 4\né 5\n";

constexpr int32_t kVocabSize = 6;

class HistoryScorerFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions & /*common*/,
                                 const std::string & /*extra*/) const override {
    return std::make_unique<HistoryScorer>(ScorerUnit::kCharacter, -0.05f);
  }
};

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

TEST(CtcSearchVocabulary, TakesTextWordStartsAndSpecialsFromTheTokenTable) {
  SymbolTable symbol_table(kTokens, false);
  auto vocabulary = MakeCtcSearchVocabulary(symbol_table, 0);
  ASSERT_EQ(vocabulary->tokens.size(), 6u);
  EXPECT_EQ(vocabulary->blank_id, 0);
  EXPECT_FALSE(vocabulary->tokens[0].emit);

  EXPECT_TRUE(vocabulary->tokens[1].emit);
  EXPECT_TRUE(vocabulary->tokens[1].starts_word);
  EXPECT_EQ(vocabulary->tokens[1].chars, std::vector<UnitId>({'t', 'h', 'e'}));

  EXPECT_FALSE(vocabulary->tokens[2].starts_word);
  EXPECT_EQ(vocabulary->tokens[2].chars, std::vector<UnitId>({'r', 'e'}));

  EXPECT_TRUE(vocabulary->tokens[3].starts_word);
  EXPECT_TRUE(vocabulary->tokens[3].chars.empty());

  EXPECT_FALSE(vocabulary->tokens[4].emit);
  EXPECT_EQ(vocabulary->tokens[5].chars, std::vector<UnitId>({0xe9}));

  // Lowercase for the scorers, beyond ASCII: Ạ (U+1EA0) becomes ạ.
  SymbolTable capitals("<blk> 0\n▁THẠ 1\n", false);
  auto lowered = MakeCtcSearchVocabulary(capitals, 0, true);
  EXPECT_EQ(lowered->tokens[1].chars, std::vector<UnitId>({'t', 'h', 0x1ea1}));
  EXPECT_TRUE(lowered->tokens[1].starts_word);
  auto kept = MakeCtcSearchVocabulary(capitals, 0);
  EXPECT_EQ(kept->tokens[1].chars, std::vector<UnitId>({'T', 'H', 0x1ea0}));
}

TEST(CtcSearchVocabulary, SilenceTokenHasNoTextAndJoinMarkersAreKept) {
  SymbolTable symbol_table("<blk> 0\nSIL 1\n##ing 2\n| 3\n", false);
  auto vocabulary = MakeCtcSearchVocabulary(symbol_table, 0);
  EXPECT_TRUE(vocabulary->tokens[1].emit);
  EXPECT_TRUE(vocabulary->tokens[1].chars.empty());
  EXPECT_FALSE(vocabulary->tokens[1].starts_word);
  // Markers the recognizer does not remove are scored as written.
  EXPECT_EQ(vocabulary->tokens[2].chars,
            std::vector<UnitId>({'#', '#', 'i', 'n', 'g'}));
  EXPECT_EQ(vocabulary->tokens[3].chars, std::vector<UnitId>({'|'}));
}

TEST(OfflineCtcPrefixBeamSearchDecoder, BatchMatchesDecodingEachUtterance) {
  RegisterScorerFactory("test-history",
                        std::make_unique<HistoryScorerFactory>());
  SymbolTable symbol_table(kTokens, false);

  FusionConfig config;
  config.scorers.resize(1);
  config.scorers[0].type = "test-history";
  config.scorers[0].weight = 0.5f;
  config.length_bonus = 0.2f;
  config.length_bonus_unit = "word";
  ASSERT_TRUE(config.Validate());
  const int32_t max_active = 4;

  const int32_t batch = 3, num_frames = 9;
  std::array<int64_t, 3> lengths = {9, 4, 6};
  auto log_probs = RandomLogProbs(batch, num_frames, 5);

  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::array<int64_t, 3> shape = {batch, num_frames, kVocabSize};
  std::array<int64_t, 1> length_shape = {batch};
  OfflineCtcPrefixBeamSearchDecoder decoder(config, max_active, symbol_table,
                                            0);
  auto results = decoder.Decode(
      Ort::Value::CreateTensor<float>(memory, log_probs.data(),
                                      log_probs.size(), shape.data(), 3),
      Ort::Value::CreateTensor<int64_t>(memory, lengths.data(), lengths.size(),
                                        length_shape.data(), 1));
  ASSERT_EQ(results.size(), 3u);

  HistoryScorer scorer(ScorerUnit::kCharacter, -0.05f);
  auto vocabulary = MakeCtcSearchVocabulary(symbol_table, 0);
  for (int32_t b = 0; b != batch; ++b) {
    CtcPrefixSearch search(MakeCtcPrefixSearchOptions(config, max_active),
                           vocabulary, {&scorer}, {0.5f});
    search.Decode(log_probs.data() + b * num_frames * kVocabSize, lengths[b]);
    auto best = search.Finish(1);
    ASSERT_EQ(best.size(), 1u);
    std::vector<int64_t> tokens(best[0].tokens.begin(), best[0].tokens.end());
    EXPECT_EQ(results[b].tokens, tokens) << "utterance " << b;
    EXPECT_EQ(results[b].timestamps, best[0].timestamps) << "utterance " << b;
    for (int64_t token : results[b].tokens) {
      EXPECT_NE(token, 0);
      EXPECT_NE(token, 4);  // <unk> is never emitted.
    }
  }
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

// Frames past an utterance's length are padding: whatever they hold, the
// result is the same. Each poison would change or break a search that read
// one padded frame.
TEST(OfflineCtcPrefixBeamSearchDecoder, PaddedFramesAreNeverRead) {
  SymbolTable symbol_table(kTokens, false);
  const int32_t batch = 3, num_frames = 9, max_active = 4;
  std::array<int64_t, 3> lengths = {9, 4, 6};
  std::array<int64_t, 3> shape = {batch, num_frames, kVocabSize};
  std::array<int64_t, 1> length_shape = {batch};
  auto memory =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  for (const char *alignment : {"sum", "max"}) {
    FusionConfig config;
    config.ctc_alignment = alignment;
    config.length_bonus = 0.2f;
    OfflineCtcPrefixBeamSearchDecoder decoder(config, max_active, symbol_table,
                                              0);
    auto decode = [&](std::vector<float> log_probs) {
      return decoder.Decode(
          Ort::Value::CreateTensor<float>(memory, log_probs.data(),
                                          log_probs.size(), shape.data(), 3),
          Ort::Value::CreateTensor<int64_t>(
              memory, lengths.data(), lengths.size(), length_shape.data(), 1));
    };
    auto clean = RandomLogProbs(batch, num_frames, 11);
    auto expected = decode(clean);

    const float poisons[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(), 1e30f,
                             0.0f};
    for (float poison : poisons) {
      auto poisoned = clean;
      for (int32_t b = 0; b != batch; ++b) {
        for (int32_t t = static_cast<int32_t>(lengths[b]); t != num_frames;
             ++t) {
          for (int32_t v = 0; v != kVocabSize; ++v) {
            poisoned[(b * num_frames + t) * kVocabSize + v] = poison;
          }
        }
      }
      auto results = decode(poisoned);
      ASSERT_EQ(results.size(), expected.size());
      for (int32_t b = 0; b != batch; ++b) {
        EXPECT_EQ(results[b].tokens, expected[b].tokens)
            << alignment << " utterance " << b << " poison " << poison;
        EXPECT_EQ(results[b].timestamps, expected[b].timestamps)
            << alignment << " utterance " << b << " poison " << poison;
      }
    }
  }
}

// --ctc-prior: the search on frames given offsets equals the search on
// frames with weight * log prior already subtracted from the listed tokens,
// blank and unlisted tokens left alone.
TEST(OfflineCtcPrefixBeamSearchDecoder, CtcPriorShiftsListedTokensEachFrame) {
  std::string filename = ::testing::TempDir() + "/ctc-prior-test.txt";
  {
    std::ofstream out(filename);
    // Blank's entry is ignored; token 3 is unlisted.
    out << "0 -0.1\n1 -0.5\n\n2 -3.0\n5 -6.0\n";
  }
  const float weight = 0.5f;
  SymbolTable symbol_table(kTokens, false);
  auto vocabulary = MakeCtcSearchVocabulary(symbol_table, 0);
  const int32_t num_frames = 12, max_active = 4;
  int32_t changed = 0;
  for (const char *alignment : {"sum", "max"}) {
    for (uint32_t seed = 1; seed != 9; ++seed) {
      auto log_probs = RandomLogProbs(1, num_frames, seed);
      auto shifted = log_probs;
      const float log_priors[kVocabSize] = {0, -0.5f, -3.0f, 0, 0, -6.0f};
      for (int32_t t = 0; t != num_frames; ++t) {
        for (int32_t v = 1; v != kVocabSize; ++v) {
          shifted[t * kVocabSize + v] -= weight * log_priors[v];
        }
      }

      FusionConfig config;
      config.ctc_alignment = alignment;
      auto decode = [&](const std::vector<float> &frames) {
        CtcPrefixSearch search(MakeCtcPrefixSearchOptions(config, max_active),
                               vocabulary, {}, {});
        search.Decode(frames.data(), num_frames);
        return search.Finish(1)[0];
      };
      auto plain = decode(log_probs);
      auto expected = decode(shifted);
      config.ctc_prior = filename;
      config.ctc_prior_weight = weight;
      auto with_prior = decode(log_probs);
      EXPECT_EQ(with_prior.tokens, expected.tokens) << alignment << seed;
      EXPECT_NEAR(with_prior.score, expected.score, 1e-4f) << alignment << seed;
      changed += with_prior.tokens != plain.tokens;

      config.ctc_prior_weight = 0;
      EXPECT_EQ(decode(log_probs).tokens, plain.tokens);
    }
  }
  // The prior is strong enough to change some results, so the test can fail.
  EXPECT_GT(changed, 0);

  EXPECT_THROW(LoadCtcPriorOffsets(filename + ".absent", 1.0f),
               std::runtime_error);
  {
    std::ofstream out(filename);
    out << "1 0.5\n";  // A log prior above 0.
  }
  EXPECT_THROW(LoadCtcPriorOffsets(filename, 1.0f), std::runtime_error);
}

TEST(OfflineCtcPrefixBeamSearchDecoder, UnknownScorerTypeIsAnError) {
  SymbolTable symbol_table(kTokens, false);
  FusionConfig config;
  config.scorers.resize(1);
  config.scorers[0].type = "nonesuch";
  EXPECT_THROW(OfflineCtcPrefixBeamSearchDecoder(config, 4, symbol_table, 0),
               std::runtime_error);
}

TEST(FusionConfig, RegistersScorersAndSearchOptions) {
  FusionConfig config;
  ParseOptions po("Test.");
  config.Register(&po);
  const char *argv[] = {"test",
                        "--scorer.type=kenlm",
                        "--scorer.model=lm.bin",
                        "--scorer.1.type=onnx-rnn",
                        "--length-bonus=1.5",
                        "--length-bonus-unit=word",
                        "--ctc-alignment=max",
                        "--token-candidates=5"};
  po.Read(8, argv);
  ASSERT_EQ(config.scorers.size(), 2u);
  EXPECT_EQ(config.scorers[0].type, "kenlm");
  EXPECT_EQ(config.scorers[1].type, "onnx-rnn");
  ASSERT_TRUE(config.Validate());

  CtcPrefixSearchOptions options = MakeCtcPrefixSearchOptions(config, 16);
  EXPECT_EQ(options.max_active, 16);
  EXPECT_EQ(options.max_token_candidates, 5);
  EXPECT_FLOAT_EQ(options.beam, 12.0f);
  EXPECT_EQ(options.alignment, CtcAlignmentMode::kMax);
  EXPECT_EQ(options.length_bonus_unit, LengthBonusUnit::kWord);
  EXPECT_FLOAT_EQ(options.length_bonus, 1.5f);

  config.score_beam = 0;
  EXPECT_TRUE(std::isinf(MakeCtcPrefixSearchOptions(config, 16).beam));

  config.ctc_alignment = "best";
  EXPECT_FALSE(config.Validate());
}

}  // namespace
}  // namespace sherpa_onnx

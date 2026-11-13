// sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"

#include <array>
#include <cmath>
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
                        "--ctc-token-candidates=5"};
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

  config.ctc_score_beam = 0;
  EXPECT_TRUE(std::isinf(MakeCtcPrefixSearchOptions(config, 16).beam));

  config.ctc_alignment = "best";
  EXPECT_FALSE(config.Validate());
}

}  // namespace
}  // namespace sherpa_onnx

// sherpa-onnx/csrc/scorer-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/scorer.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/option-collection.h"
#include "sherpa-onnx/csrc/option-json.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

// Creates a BigramTableScorer whose default log score is the model path
// parsed as a number, so a test can see its options arrive.
class BigramTableFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions &common,
                                 const std::string &extra) const override {
    if (extra != "{}") {
      throw std::runtime_error("test-bigram takes no extra options: " + extra);
    }
    ScorerUnit unit;
    if (!ParseScorerUnit(common.unit, &unit)) {
      throw std::runtime_error("bad unit " + common.unit);
    }
    return std::make_unique<BigramTableScorer>(unit, std::stof(common.model));
  }
};

std::vector<ScorerOptions> ReadScorers(
    const std::vector<std::string> &options) {
  std::vector<ScorerOptions> scorers;
  std::vector<const char *> argv = {"scorer-test"};
  for (const auto &option : options) {
    argv.push_back(option.c_str());
  }
  ParseOptions po("Test.");
  RegisterOptionCollection(&po, "scorer", &scorers, "Scorers.");
  po.Read(static_cast<int32_t>(argv.size()), argv.data());
  return scorers;
}

TEST(ScorerOptions, OneScorerFromUnindexedOptions) {
  auto scorers =
      ReadScorers({"--scorer.type=kenlm", "--scorer.model=lm.bin",
                   "--scorer.weight=0.6", "--scorer.unit=token",
                   "--scorer.num-threads=2", "--scorer.provider=cuda"});
  ASSERT_EQ(scorers.size(), 1u);
  EXPECT_EQ(scorers[0].type, "kenlm");
  EXPECT_EQ(scorers[0].common.model, "lm.bin");
  EXPECT_EQ(scorers[0].common.unit, "token");
  EXPECT_EQ(scorers[0].common.num_threads, 2);
  EXPECT_EQ(scorers[0].common.provider.provider, "cuda");
  EXPECT_FLOAT_EQ(scorers[0].weight, 0.6f);
  EXPECT_EQ(scorers[0].extra, "{}");
  EXPECT_TRUE(scorers[0].Validate());
}

TEST(ScorerOptions, SeveralScorersFromJsonWithExtraPassedAsText) {
  std::vector<ScorerOptions> scorers;
  ParseOptions po("Test.");
  RegisterOptionCollection(&po, "scorer", &scorers, "Scorers.");
  std::string error;
  ASSERT_TRUE(ApplyJsonOptions(
      R"({"scorer": [
            {"type": "kenlm", "model": "lm.bin", "weight": 0.6},
            {"type": "onnx-rnn", "model": "lstm.onnx", "weight": 0.3,
             "extra": {"state-layout": "gru"}}]})",
      &po, &error))
      << error;
  ASSERT_EQ(scorers.size(), 2u);
  EXPECT_EQ(scorers[1].type, "onnx-rnn");
  EXPECT_EQ(scorers[1].common.unit, "char");
  EXPECT_FLOAT_EQ(scorers[1].weight, 0.3f);
  EXPECT_EQ(scorers[1].extra, R"({"state-layout":"gru"})");
}

TEST(ScorerOptions, ValidateRejectsMissingTypeAndBadUnit) {
  ScorerOptions options;
  EXPECT_FALSE(options.Validate());
  options.type = "kenlm";
  EXPECT_TRUE(options.Validate());
  options.common.unit = "word";
  EXPECT_FALSE(options.Validate());
}

TEST(ScorerRegistry, CreatesByTypeAndReportsUnknownTypes) {
  RegisterScorerFactory("test-bigram", std::make_unique<BigramTableFactory>());
  EXPECT_THROW(RegisterScorerFactory("test-bigram",
                                     std::make_unique<BigramTableFactory>()),
               std::runtime_error);

  auto options =
      ReadScorers({"--scorer.0.type=test-bigram", "--scorer.0.model=-2.5",
                   "--scorer.1.type=test-bigram", "--scorer.1.model=-7",
                   "--scorer.1.unit=token"});
  auto scorers = CreateScorers(options);
  ASSERT_EQ(scorers.size(), 2u);
  EXPECT_EQ(scorers[0]->Info().unit, ScorerUnit::kCharacter);
  EXPECT_EQ(scorers[1]->Info().unit, ScorerUnit::kAsrToken);
  auto *second = static_cast<BigramTableScorer *>(scorers[1].get());
  EXPECT_FLOAT_EQ(second->LogScore(1, 2), -7.0f);

  options[0].type = "nonesuch";
  try {
    CreateScorers(options);
    FAIL() << "expected an unknown-type error";
  } catch (const std::runtime_error &e) {
    EXPECT_EQ(std::string(e.what()),
              "Unknown scorer type 'nonesuch'. Registered types: test-bigram");
  }
}

TEST(Scorer, ValueStateAdvancesAndRejects) {
  BigramTableScorer scorer(ScorerUnit::kAsrToken);
  scorer.Set(BigramTableScorer::kStart, 5, -1.0f);
  scorer.Set(5, 6, -0.25f);
  scorer.Reject(5, 7);
  scorer.SetFinal(6, -0.5f);
  ASSERT_EQ(scorer.Info().state_bytes, 8u);
  EXPECT_TRUE(scorer.Info().recombines);

  alignas(8) std::byte start[8], after5[8], after6[8], unwritten[8];
  scorer.Start(nullptr, start);

  ScorerExtension first = {{nullptr, start}, 5, after5};
  ScorerAdvanceResult result;
  scorer.Advance({&first, 1}, &result);
  EXPECT_FALSE(result.rejected);
  EXPECT_FLOAT_EQ(result.log_score, -1.0f);

  ScorerExtension batch[2] = {{{nullptr, after5}, 6, after6},
                              {{nullptr, after5}, 7, unwritten}};
  ScorerAdvanceResult results[2];
  scorer.Advance({batch, 2}, results);
  EXPECT_FLOAT_EQ(results[0].log_score, -0.25f);
  EXPECT_FALSE(results[0].rejected);
  EXPECT_TRUE(results[1].rejected);

  ScorerStateRef end = {nullptr, after6};
  ScorerFinalScore final_score;
  scorer.Finish({&end, 1}, &final_score);
  EXPECT_TRUE(final_score.accepted);
  EXPECT_FLOAT_EQ(final_score.log_score, -0.5f);

  // Equal histories' last units give equal bytes: what recombination compares.
  alignas(8) std::byte other6[8];
  ScorerExtension other = {{nullptr, start}, 6, other6};
  scorer.Advance({&other, 1}, &result);
  EXPECT_EQ(std::memcmp(after6, other6, 8), 0);
}

TEST(Scorer, WrappedStateIsCopiedAndDestroyedThroughInfo) {
  HistoryScorer scorer(ScorerUnit::kCharacter, -0.5f);
  const ScorerInfo &info = scorer.Info();
  ASSERT_NE(info.copy_state, nullptr);
  ASSERT_NE(info.destroy_state, nullptr);
  EXPECT_FALSE(info.recombines);
  ASSERT_EQ(HistoryScorer::LiveStates(), 0);

  auto utterance = scorer.NewUtterance();
  alignas(8) std::byte start[8], child[8], copy[8];
  scorer.Start(utterance.get(), start);

  ScorerExtension extension = {{utterance.get(), start}, 'a', child};
  ScorerAdvanceResult result;
  scorer.Advance({&extension, 1}, &result);
  EXPECT_FLOAT_EQ(result.log_score, -0.5f);
  EXPECT_EQ(HistoryScorer::HistoryOf(child), std::vector<UnitId>({'a'}));
  EXPECT_TRUE(HistoryScorer::HistoryOf(start).empty());

  info.copy_state(child, copy);
  EXPECT_EQ(HistoryScorer::LiveStates(), 3);
  EXPECT_EQ(HistoryScorer::HistoryOf(copy), std::vector<UnitId>({'a'}));
  EXPECT_EQ(static_cast<HistoryScorer::Utterance *>(utterance.get())->advances,
            1);

  info.destroy_state(start);
  info.destroy_state(child);
  info.destroy_state(copy);
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

}  // namespace
}  // namespace sherpa_onnx

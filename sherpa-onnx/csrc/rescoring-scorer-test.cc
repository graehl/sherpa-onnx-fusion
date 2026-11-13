// sherpa-onnx/csrc/rescoring-scorer-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/rescoring-scorer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/ctc-prefix-search.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

std::vector<float> RandomLogProbs(int32_t num_frames, int32_t vocab_size,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-3.0f, 3.0f);
  std::vector<float> log_probs(num_frames * vocab_size);
  for (int32_t t = 0; t != num_frames; ++t) {
    float *row = log_probs.data() + t * vocab_size;
    float total = 0;
    for (int32_t v = 0; v != vocab_size; ++v) {
      row[v] = uniform(rng);
      total += std::exp(row[v]);
    }
    for (int32_t v = 0; v != vocab_size; ++v) row[v] -= std::log(total);
  }
  return log_probs;
}

std::shared_ptr<CtcSearchVocabulary> Vocabulary(int32_t vocab_size) {
  auto vocabulary = std::make_shared<CtcSearchVocabulary>();
  vocabulary->tokens.resize(vocab_size);
  for (int32_t v = 1; v < vocab_size; ++v) {
    vocabulary->tokens[v].chars = {'a' + v - 1};
  }
  return vocabulary;
}

std::unique_ptr<BigramTableScorer> Bigram() {
  auto bigram =
      std::make_unique<BigramTableScorer>(ScorerUnit::kAsrToken, -2.0f);
  bigram->Set(BigramTableScorer::kStart, 2, -0.2f);
  bigram->Set(2, 3, -0.1f);
  bigram->Set(3, 1, -0.4f);
  bigram->SetFinal(1, -0.3f);
  bigram->SetFinal(3, -1.1f);
  return bigram;
}

double BigramScore(const BigramTableScorer &bigram,
                   const std::vector<int32_t> &tokens) {
  double score = 0;
  UnitId previous = BigramTableScorer::kStart;
  for (int32_t token : tokens) {
    score += bigram.LogScore(previous, token);
    previous = token;
  }
  return score + bigram.FinalLogScore(previous);
}

// The search with a rescoring scorer keeps exactly the hypotheses the search
// without a scorer keeps, and ranks them by their score plus the weighted
// scorer score of each whole output.
TEST(RescoringScorer, RanksTheUnscoredSearchsSurvivorsByTheWholeOutput) {
  const int32_t num_frames = 12, vocab_size = 5, max_active = 3;
  const float weight = 1.5f;
  auto reference = Bigram();
  int32_t reordered = 0;
  for (uint32_t seed = 0; seed != 40; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    CtcPrefixSearchOptions options;
    options.max_active = max_active;
    options.max_token_candidates = vocab_size;

    CtcPrefixSearch plain(options, Vocabulary(vocab_size), {}, {});
    plain.Decode(log_probs.data(), num_frames);
    auto unscored = plain.Finish(max_active);

    std::vector<std::pair<double, std::vector<int32_t>>> expected;
    for (const auto &hypothesis : unscored) {
      expected.push_back(
          {hypothesis.score +
               weight * BigramScore(*reference, hypothesis.tokens),
           hypothesis.tokens});
    }
    std::stable_sort(
        expected.begin(), expected.end(),
        [](const auto &a, const auto &b) { return a.first > b.first; });

    auto rescoring = MakeRescoringScorer(Bigram());
    {
      CtcPrefixSearch search(options, Vocabulary(vocab_size), {rescoring.get()},
                             {weight});
      search.Decode(log_probs.data(), num_frames);
      auto rescored = search.Finish(max_active);
      ASSERT_EQ(rescored.size(), expected.size()) << "seed " << seed;
      for (size_t i = 0; i != rescored.size(); ++i) {
        EXPECT_EQ(rescored[i].tokens, expected[i].second) << "seed " << seed;
        EXPECT_NEAR(rescored[i].score, expected[i].first, 1e-4);
        EXPECT_NEAR(rescored[i].scorer_log_scores[0],
                    BigramScore(*reference, rescored[i].tokens), 1e-4);
      }
      reordered += rescored[0].tokens != unscored[0].tokens;
    }
    EXPECT_EQ(RescoringScorerLiveHistories(), 0) << "seed " << seed;
  }
  // The scorer changes some winners, or the test shows nothing.
  EXPECT_GT(reordered, 3);
}

// A continuation the wrapped scorer rejects is found at the end.
TEST(RescoringScorer, ReportsAWrappedRejectionAtTheEnd) {
  auto bigram = Bigram();
  bigram->Reject(2, 3);
  auto rescoring = MakeRescoringScorer(std::move(bigram));
  const ScorerInfo &info = rescoring->Info();
  auto utterance = rescoring->NewUtterance();
  alignas(8) std::byte start[8], two[8], two_three[8], two_one[8];
  rescoring->Start(utterance.get(), start);
  ScorerExtension first = {{utterance.get(), start}, 2, two};
  ScorerAdvanceResult result;
  rescoring->Advance({&first, 1}, &result);
  EXPECT_EQ(result.log_score, 0.0f);
  ScorerExtension second[2] = {{{utterance.get(), two}, 3, two_three},
                               {{utterance.get(), two}, 1, two_one}};
  ScorerAdvanceResult results[2];
  rescoring->Advance({second, 2}, results);
  EXPECT_FALSE(results[0].rejected);

  ScorerStateRef ends[2] = {{utterance.get(), two_three},
                            {utterance.get(), two_one}};
  ScorerFinalScore finals[2];
  rescoring->Finish({ends, 2}, finals);
  EXPECT_FALSE(finals[0].accepted);
  EXPECT_TRUE(finals[1].accepted);

  for (std::byte *state : {start, two, two_three, two_one}) {
    info.destroy_state(state);
  }
  EXPECT_EQ(RescoringScorerLiveHistories(), 0);
}

}  // namespace
}  // namespace sherpa_onnx

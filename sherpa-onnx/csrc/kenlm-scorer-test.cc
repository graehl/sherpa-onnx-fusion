// sherpa-onnx/csrc/kenlm-scorer-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/kenlm-scorer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

constexpr float kLn10 = 2.302585092994046f;

// A character bigram model over a, b, and the space word "▁". Values are
// log10 probabilities and backoffs.
const char kArpa[] = R"(\data\
ngram 1=6
ngram 2=4

\1-grams:
-1.0	<unk>
-99	<s>	-0.5
-0.7	</s>
-0.6	a	-0.3
-0.8	b	-0.2
-0.9	▁	-0.1

\2-grams:
-0.2	<s> a
-0.1	a b
-0.4	b ▁
-0.3	b </s>

\end\
)";

class KenLmScorerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    filename_ = ::testing::TempDir() + "kenlm-scorer-test.arpa";
    std::ofstream os(filename_);
    os << kArpa;
  }
  void TearDown() override { std::remove(filename_.c_str()); }

  std::unique_ptr<Scorer> Create(const std::string &extra = "{}",
                                 const std::string &unit = "char") {
    CommonScorerOptions common;
    common.model = filename_;
    common.unit = unit;
    return MakeKenLmScorerFactory()->Create(common, extra);
  }

  std::string filename_;
};

// Advances from the start through "units"; returns each unit's score and
// leaves the final state in *state.
std::vector<float> Scores(Scorer *scorer, const std::vector<UnitId> &units,
                          std::vector<std::byte> *state) {
  size_t size = scorer->Info().state_bytes;
  std::vector<std::byte> current(size), next(size);
  scorer->Start(nullptr, current.data());
  std::vector<float> scores;
  for (UnitId unit : units) {
    ScorerExtension extension = {{nullptr, current.data()}, unit, next.data()};
    ScorerAdvanceResult result;
    scorer->Advance({&extension, 1}, &result);
    EXPECT_FALSE(result.rejected);
    scores.push_back(result.log_score);
    current.swap(next);
  }
  *state = current;
  return scores;
}

TEST_F(KenLmScorerTest, ScoresCharactersBoundariesAndEndOfSentence) {
  auto scorer = Create();
  EXPECT_EQ(scorer->Info().unit, ScorerUnit::kCharacter);
  EXPECT_TRUE(scorer->Info().recombines);

  std::vector<std::byte> state;
  // "ab " then a: <s> a, a b, b ▁ are bigrams; ▁ a backs off from ▁.
  auto scores = Scores(scorer.get(), {'a', 'b', ' ', 'a'}, &state);
  ASSERT_EQ(scores.size(), 4u);
  EXPECT_NEAR(scores[0], -0.2f * kLn10, 1e-5);
  EXPECT_NEAR(scores[1], -0.1f * kLn10, 1e-5);
  EXPECT_NEAR(scores[2], -0.4f * kLn10, 1e-5);
  EXPECT_NEAR(scores[3], (-0.1f - 0.6f) * kLn10, 1e-5);

  // After "ab": b </s> is a bigram.
  Scores(scorer.get(), {'a', 'b'}, &state);
  ScorerStateRef end = {nullptr, state.data()};
  ScorerFinalScore final_score;
  scorer->Finish({&end, 1}, &final_score);
  EXPECT_TRUE(final_score.accepted);
  EXPECT_NEAR(final_score.log_score, -0.3f * kLn10, 1e-5);
}

TEST_F(KenLmScorerTest, UnknownCharacterScoresAsUnk) {
  auto scorer = Create();
  std::vector<std::byte> state;
  auto scores = Scores(scorer.get(), {'a', 'z'}, &state);
  // a z: backoff of a, then <unk>.
  EXPECT_NEAR(scores[1], (-0.3f - 1.0f) * kLn10, 1e-5);
}

TEST_F(KenLmScorerTest, EqualContextsHaveEqualStateBytes) {
  auto scorer = Create();
  std::vector<std::byte> after_ab, after_bab, after_ba;
  Scores(scorer.get(), {'a', 'b'}, &after_ab);
  Scores(scorer.get(), {'b', 'a', 'b'}, &after_bab);
  Scores(scorer.get(), {'b', 'a'}, &after_ba);
  EXPECT_EQ(after_ab, after_bab);
  EXPECT_NE(after_ab, after_ba);

  // Histories that back off to the same context: nothing follows "▁" or
  // "<unk>" in a bigram, so both states are the empty context.
  std::vector<std::byte> after_space, after_unknown;
  Scores(scorer.get(), {'a', 'z'}, &after_unknown);
  Scores(scorer.get(), {'b', 'z'}, &after_space);
  EXPECT_EQ(after_unknown, after_space);
}

TEST_F(KenLmScorerTest, SpaceWordIsConfigurable) {
  CommonScorerOptions common;
  common.model = filename_;
  common.space = "b";
  auto scorer = MakeKenLmScorerFactory()->Create(common, "{}");
  std::vector<std::byte> state;
  auto scores = Scores(scorer.get(), {'a', ' '}, &state);
  EXPECT_NEAR(scores[1], -0.1f * kLn10, 1e-5);
}

TEST_F(KenLmScorerTest, RejectsOptionsItDoesNotUnderstand) {
  EXPECT_THROW(Create(R"({"nonesuch": 1})"), std::runtime_error);
  EXPECT_THROW(Create("[]"), std::runtime_error);
  EXPECT_THROW(Create("{}", "token"), std::runtime_error);
  CommonScorerOptions common;
  common.model = filename_ + ".missing";
  EXPECT_THROW(MakeKenLmScorerFactory()->Create(common, "{}"),
               std::runtime_error);
}

}  // namespace
}  // namespace sherpa_onnx

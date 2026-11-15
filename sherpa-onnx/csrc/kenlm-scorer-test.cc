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

  // Units given ahead by a decoder, and one that was not, score the same.
  auto prepared = Create();
  std::vector<UnitId> units = {'a', ' '};
  prepared->Prepare({units.data(), units.size()});
  EXPECT_EQ(Scores(prepared.get(), {'a', 'b', ' ', 'a'}, &state), scores);

  // A floor bounds each unit from below: only the backed-off a is under it.
  auto floored = Create(R"({"floor": -1.0})");
  auto floored_scores = Scores(floored.get(), {'a', 'b', ' ', 'a'}, &state);
  EXPECT_NEAR(floored_scores[0], -0.2f * kLn10, 1e-5);
  EXPECT_FLOAT_EQ(floored_scores[3], -1.0f);

  // After "ab": b </s> is a bigram.
  Scores(scorer.get(), {'a', 'b'}, &state);
  ScorerStateRef end = {nullptr, state.data()};
  ScorerFinalScore final_score;
  scorer->Finish({&end, 1}, &final_score);
  EXPECT_TRUE(final_score.accepted);
  EXPECT_NEAR(final_score.log_score, -0.3f * kLn10, 1e-5);
}

// "words": the model's words are words. Characters cost nothing until the
// boundary, where the word is scored; an open word is scored at the end.
TEST_F(KenLmScorerTest, WordModelScoresEachWordAtItsBoundary) {
  auto scorer = Create(R"({"words": true})");
  EXPECT_TRUE(scorer->Info().recombines);
  auto utterance = scorer->NewUtterance();
  ASSERT_NE(utterance, nullptr);
  size_t size = scorer->Info().state_bytes;
  std::vector<std::byte> current(size), next(size);
  scorer->Start(utterance.get(), current.data());
  std::vector<float> scores;
  // "a b ba": <s> a and a b are bigrams; "ba" is unknown, after b.
  for (UnitId unit : {'a', ' ', 'b', ' ', ' ', 'b', 'a'}) {
    ScorerExtension extension = {
        {utterance.get(), current.data()}, unit, next.data()};
    ScorerAdvanceResult result;
    scorer->Advance({&extension, 1}, &result);
    scores.push_back(result.log_score);
    current.swap(next);
  }
  EXPECT_EQ(scores[0], 0);
  EXPECT_NEAR(scores[1], -0.2f * kLn10, 1e-5);
  EXPECT_EQ(scores[2], 0);
  EXPECT_NEAR(scores[3], -0.1f * kLn10, 1e-5);
  EXPECT_EQ(scores[4], 0);  // A second boundary closes nothing.
  EXPECT_EQ(scores[5], 0);
  EXPECT_EQ(scores[6], 0);

  ScorerStateRef end = {utterance.get(), current.data()};
  ScorerFinalScore final_score;
  scorer->Finish({&end, 1}, &final_score);
  // b backs off to <unk>, then </s> after <unk>.
  EXPECT_NEAR(final_score.log_score, (-0.2f - 1.0f - 0.7f) * kLn10, 1e-5);

  // With a floor the unknown word costs no more than the floor.
  auto floored = Create(R"({"words": true, "floor": -1.5})");
  auto floored_utterance = floored->NewUtterance();
  std::vector<std::byte> open(size), closed(size);
  floored->Start(floored_utterance.get(), open.data());
  for (UnitId unit : {'b', 'a', ' '}) {
    ScorerExtension extension = {
        {floored_utterance.get(), open.data()}, unit, closed.data()};
    ScorerAdvanceResult result;
    floored->Advance({&extension, 1}, &result);
    if (unit == ' ') EXPECT_FLOAT_EQ(result.log_score, -1.5f);
    open.swap(closed);
  }

  // The same partial word reached twice is the same state.
  std::vector<std::byte> first(size), second(size), start(size);
  scorer->Start(utterance.get(), start.data());
  for (std::vector<std::byte> *out : {&first, &second}) {
    ScorerExtension extension = {
        {utterance.get(), start.data()}, 'b', out->data()};
    ScorerAdvanceResult result;
    scorer->Advance({&extension, 1}, &result);
  }
  EXPECT_EQ(first, second);
}

// A model over the recognizer's tokens: the table's symbols are its words.
TEST_F(KenLmScorerTest, TokenUnitsUseTheTokenTablesSymbols) {
  std::string tokens = ::testing::TempDir() + "kenlm-scorer-test.tokens";
  {
    std::ofstream os(tokens);
    os << "<blk> 0\nb 1\na 2\nzz 3\n";
  }
  auto scorer = Create("{\"tokens\": \"" + tokens + "\"}", "token");
  EXPECT_EQ(scorer->Info().unit, ScorerUnit::kAsrToken);
  std::vector<std::byte> state;
  // Token 2 is "a" and token 1 is "b": the same bigrams as the characters.
  auto scores = Scores(scorer.get(), {2, 1}, &state);
  EXPECT_NEAR(scores[0], -0.2f * kLn10, 1e-5);
  EXPECT_NEAR(scores[1], -0.1f * kLn10, 1e-5);
  // A token the model lacks and one past the table are both <unk>.
  auto unknown = Scores(scorer.get(), {3}, &state);
  auto beyond = Scores(scorer.get(), {40}, &state);
  EXPECT_FLOAT_EQ(unknown[0], beyond[0]);
  EXPECT_THROW(Create("{}", "token"), std::runtime_error);
  std::remove(tokens.c_str());

  // "id-offset": the model's words are IDs; this model has none of them.
  auto by_id = Create("{\"id-offset\": 2}", "token");
  EXPECT_EQ(by_id->Info().unit, ScorerUnit::kAsrToken);
  auto ids = Scores(by_id.get(), {2}, &state);
  EXPECT_FLOAT_EQ(ids[0], beyond[0]);

  // A model whose words are "t" and an ID: with the prefix and an offset
  // of 10, tokens 1 and 2 are "t11" and "t12".
  std::string ids_arpa = ::testing::TempDir() + "kenlm-scorer-test-ids.arpa";
  {
    std::ofstream os(ids_arpa);
    os << "\\data\\\nngram 1=5\nngram 2=2\n\n\\1-grams:\n-1.0\t<unk>\n"
          "-99\t<s>\t-0.5\n-0.7\t</s>\n-0.6\tt11\t-0.3\n-0.8\tt12\t-0.2\n\n"
          "\\2-grams:\n-0.2\t<s> t11\n-0.1\tt11 t12\n\n\\end\\\n";
  }
  CommonScorerOptions common;
  common.model = ids_arpa;
  common.unit = "token";
  auto prefixed = MakeKenLmScorerFactory()->Create(
      common, "{\"id-prefix\": \"t\", \"id-offset\": 10}");
  auto by_prefix = Scores(prefixed.get(), {1, 2}, &state);
  EXPECT_NEAR(by_prefix[0], -0.2f * kLn10, 1e-5);
  EXPECT_NEAR(by_prefix[1], -0.1f * kLn10, 1e-5);
  std::remove(ids_arpa.c_str());
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

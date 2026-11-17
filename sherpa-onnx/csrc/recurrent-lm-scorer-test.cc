// sherpa-onnx/csrc/recurrent-lm-scorer-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/recurrent-lm-scorer.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>   // NOLINT
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/ctc-prefix-search.h"

namespace sherpa_onnx {
namespace {

constexpr int32_t kVocab = 5;  // 0 boundary, 1 unknown, 2 'a', 3 'b', 4 space

// A deterministic stand-in for a two-tensor (LSTM-shaped) model with
// [layers 2, hidden 3] and [layers 1, hidden 2] states. Each new state
// element is 0.5 * old + (unit + 1) * 0.1 * (index + 1), and the log
// probability of next unit v is -(0.3 * (v + 1) * (1 + first state's last
// layer, summed)), unnormalized, which is enough to check plumbing.
class FakeStep : public RecurrentLmStep {
 public:
  struct Shape {
    int32_t layers;
    int32_t hidden;
  };

  // Each state [layers, batch, hidden], and [batch, vocab] log probabilities.
  struct FakeResult : public Result {
    int32_t batch = 0;
    std::vector<std::vector<float>> states;
    std::vector<float> log_probs;
    const float *LogProbs(int32_t item) const override {
      return log_probs.data() + item * kVocab;
    }
  };

  int32_t VocabSize() const override { return kVocab; }

  std::shared_ptr<const Result> Run(
      const std::vector<int64_t> &units,
      const std::vector<Parent> &parents) const override {
    size_t batch = units.size();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      batch_sizes.push_back(static_cast<int32_t>(batch));
    }
    auto result = std::make_shared<FakeResult>();
    result->batch = static_cast<int32_t>(batch);
    std::vector<std::vector<float>> &out_ = result->states;
    std::vector<float> &log_probs_ = result->log_probs;
    out_.resize(shapes_.size());
    for (size_t k = 0; k != shapes_.size(); ++k) {
      int32_t layers = shapes_[k].layers, hidden = shapes_[k].hidden;
      out_[k].resize(layers * batch * hidden);
      for (int32_t layer = 0; layer != layers; ++layer) {
        for (size_t i = 0; i != batch; ++i) {
          const auto *from = static_cast<const FakeResult *>(parents[i].result);
          for (int32_t h = 0; h != hidden; ++h) {
            float in = 0;
            if (from != nullptr) {
              in = from->states[k][(layer * from->batch + parents[i].item) *
                                       hidden +
                                   h];
            }
            out_[k][(layer * batch + i) * hidden + h] =
                0.5f * in + (units[i] + 1) * 0.1f * (layer * hidden + h + 1);
          }
        }
      }
    }
    log_probs_.resize(batch * kVocab);
    for (size_t i = 0; i != batch; ++i) {
      float sum = 0;
      for (int32_t h = 0; h != shapes_[0].hidden; ++h) {
        sum += out_[0][(1 * batch + i) * shapes_[0].hidden + h];
      }
      for (int32_t v = 0; v != kVocab; ++v) {
        log_probs_[i * kVocab + v] = -0.3f * (v + 1) * (1 + sum);
      }
    }
    return result;
  }

  mutable std::vector<int32_t> batch_sizes;  // One entry per Run.

 private:
  std::vector<Shape> shapes_ = {{2, 3}, {1, 2}};
  mutable std::mutex mutex_;  // For batch_sizes.
};

// The same model stepped one history at a time: the log probability of each
// next unit after "units" (model IDs), starting from the boundary.
std::vector<float> ReferenceLogProbs(const std::vector<int32_t> &units) {
  FakeStep step;
  std::shared_ptr<const RecurrentLmStep::Result> state;  // the zero state
  std::vector<int32_t> history = {0};
  history.insert(history.end(), units.begin(), units.end());
  std::vector<float> result;
  for (int32_t unit : history) {
    state = step.Run({unit}, {{state.get(), 0}});
    result.assign(state->LogProbs(0), state->LogProbs(0) + kVocab);
  }
  return result;
}

RecurrentLmVocabulary Vocabulary() {
  RecurrentLmVocabulary vocabulary;
  vocabulary.id_of_char.Set('a', 2);
  vocabulary.id_of_char.Set('b', 3);
  vocabulary.id_of_char.Set(' ', 4);
  return vocabulary;
}

struct Fixture {
  explicit Fixture(int32_t max_batch = 0) {
    auto owned = std::make_unique<FakeStep>();
    step = owned.get();
    RecurrentLmVocabulary vocabulary = Vocabulary();
    vocabulary.max_batch = max_batch;
    scorer = MakeRecurrentLmScorer(std::move(owned), std::move(vocabulary));
    utterance = scorer->NewUtterance();
  }
  FakeStep *step;
  std::unique_ptr<Scorer> scorer;
  std::unique_ptr<ScorerUtterance> utterance;
};

// One scorer serves several threads at once, each with its own utterance:
// every thread gets the scores of stepping its history alone.
TEST(RecurrentLmScorer, ThreadsShareOneScorer) {
  Fixture f;
  Scorer *scorer = f.scorer.get();
  const int32_t kThreads = 8, kRounds = 200;
  std::vector<int32_t> mismatches(kThreads, 0);
  std::vector<std::thread> threads;
  for (int32_t t = 0; t != kThreads; ++t) {
    threads.emplace_back([scorer, t, &mismatches]() {
      // Thread t spells a different history: t % 3 'a's, then 'b', then 'a'.
      std::vector<UnitId> spelling(t % 3, 'a');
      spelling.push_back('b');
      spelling.push_back('a');
      std::vector<float> expected;
      std::vector<int32_t> ids;
      for (UnitId unit : spelling) {
        int32_t id = unit == 'a' ? 2 : 3;
        expected.push_back(ReferenceLogProbs(ids)[id]);
        ids.push_back(id);
      }
      for (int32_t round = 0; round != kRounds; ++round) {
        auto utterance = scorer->NewUtterance();
        alignas(8) std::byte current[8], next[8];
        scorer->Start(utterance.get(), current);
        for (size_t i = 0; i != spelling.size(); ++i) {
          ScorerExtension extension = {
              {utterance.get(), current}, spelling[i], next};
          ScorerAdvanceResult result;
          scorer->Advance({&extension, 1}, &result);
          if (result.log_score != expected[i]) ++mismatches[t];
          scorer->Info().destroy_state(current);
          std::memcpy(current, next, sizeof(current));
        }
        scorer->Info().destroy_state(current);
      }
    });
  }
  for (auto &thread : threads) thread.join();
  for (int32_t t = 0; t != kThreads; ++t) EXPECT_EQ(mismatches[t], 0) << t;
  EXPECT_EQ(RecurrentLmLiveNodes(), 0);
}

TEST(RecurrentLmScorer, ScoresMatchSteppingOneHistoryAtATime) {
  ASSERT_EQ(RecurrentLmLiveNodes(), 0);
  {
    Fixture f;
    const ScorerInfo &info = f.scorer->Info();
    alignas(8) std::byte start[8], a[8], ab[8], ax[8];
    f.scorer->Start(f.utterance.get(), start);

    ScorerExtension first = {{f.utterance.get(), start}, 'a', a};
    ScorerAdvanceResult result;
    f.scorer->Advance({&first, 1}, &result);
    EXPECT_FLOAT_EQ(result.log_score, ReferenceLogProbs({})[2]);

    // Two extensions of the same parent, one with an unknown character.
    ScorerExtension second[2] = {{{f.utterance.get(), a}, 'b', ab},
                                 {{f.utterance.get(), a}, 'x', ax}};
    ScorerAdvanceResult results[2];
    f.scorer->Advance({second, 2}, results);
    EXPECT_FLOAT_EQ(results[0].log_score, ReferenceLogProbs({2})[3]);
    EXPECT_FLOAT_EQ(results[1].log_score, ReferenceLogProbs({2})[1]);

    ScorerStateRef ends[2] = {{f.utterance.get(), ab}, {f.utterance.get(), ax}};
    ScorerFinalScore finals[2];
    f.scorer->Finish({ends, 2}, finals);
    EXPECT_FLOAT_EQ(finals[0].log_score, ReferenceLogProbs({2, 3})[0]);
    EXPECT_FLOAT_EQ(finals[1].log_score, ReferenceLogProbs({2, 1})[0]);

    for (std::byte *state : {start, a, ab, ax}) info.destroy_state(state);
    EXPECT_EQ(RecurrentLmLiveNodes(), 0);
  }
}

TEST(RecurrentLmScorer, FloorCapsUnitAndEndScores) {
  auto vocabulary = Vocabulary();
  vocabulary.score_floor = -1.0f;
  auto scorer = MakeRecurrentLmScorer(std::make_unique<FakeStep>(), vocabulary);
  auto utterance = scorer->NewUtterance();
  alignas(8) std::byte start[8], a[8], ax[8];
  scorer->Start(utterance.get(), start);
  ScorerExtension first = {{utterance.get(), start}, 'a', a};
  ScorerAdvanceResult result;
  scorer->Advance({&first, 1}, &result);
  ASSERT_LT(ReferenceLogProbs({})[2], -1.0f);
  EXPECT_FLOAT_EQ(result.log_score, -1.0f);
  ScorerExtension unknown = {{utterance.get(), a}, 'x', ax};
  scorer->Advance({&unknown, 1}, &result);
  EXPECT_FLOAT_EQ(result.log_score, -1.0f);
  ScorerStateRef ends[2] = {{utterance.get(), start}, {utterance.get(), ax}};
  ScorerFinalScore finals[2];
  scorer->Finish({ends, 2}, finals);
  ASSERT_GT(ReferenceLogProbs({})[0], -1.0f);
  EXPECT_FLOAT_EQ(finals[0].log_score, ReferenceLogProbs({})[0]);
  ASSERT_LT(ReferenceLogProbs({2, 1})[0], -1.0f);
  EXPECT_FLOAT_EQ(finals[1].log_score, -1.0f);
  for (std::byte *state : {start, a, ax}) scorer->Info().destroy_state(state);
  EXPECT_EQ(RecurrentLmLiveNodes(), 0);
}

TEST(RecurrentLmScorer, FlooredUnitPreservesTheRecurrentHistory) {
  auto vocabulary = Vocabulary();
  vocabulary.score_floor = -2.0f;
  auto scorer = MakeRecurrentLmScorer(std::make_unique<FakeStep>(), vocabulary);
  auto utterance = scorer->NewUtterance();
  alignas(8) std::byte start[8], a[8];
  scorer->Start(utterance.get(), start);
  ScorerExtension extension = {{utterance.get(), start}, 'a', a};
  ScorerAdvanceResult result;
  scorer->Advance({&extension, 1}, &result);
  ASSERT_LT(ReferenceLogProbs({})[2], -2.0f);
  EXPECT_FLOAT_EQ(result.log_score, -2.0f);
  ScorerStateRef end = {utterance.get(), a};
  ScorerFinalScore final;
  scorer->Finish({&end, 1}, &final);
  ASSERT_GT(ReferenceLogProbs({2})[0], -2.0f);
  EXPECT_FLOAT_EQ(final.log_score, ReferenceLogProbs({2})[0]);
  scorer->Info().destroy_state(start);
  scorer->Info().destroy_state(a);
}

// A model trained on the recognizer's tokens: unit IDs are token IDs.
TEST(RecurrentLmScorer, TokenUnitsAreTheModelsOwnIds) {
  RecurrentLmVocabulary vocabulary;
  vocabulary.token_units = true;
  vocabulary.boundary_id = 0;  // What ReferenceLogProbs starts from.
  vocabulary.unknown_id = 1;
  auto scorer = MakeRecurrentLmScorer(std::make_unique<FakeStep>(), vocabulary);
  const ScorerInfo &info = scorer->Info();
  EXPECT_EQ(info.unit, ScorerUnit::kAsrToken);
  auto utterance = scorer->NewUtterance();
  alignas(8) std::byte start[8], three[8], three_two[8], beyond[8];
  scorer->Start(utterance.get(), start);

  ScorerExtension first = {{utterance.get(), start}, 3, three};
  ScorerAdvanceResult result;
  scorer->Advance({&first, 1}, &result);
  EXPECT_FLOAT_EQ(result.log_score, ReferenceLogProbs({})[3]);

  // A token ID the model does not score counts as the unknown unit.
  ScorerExtension second[2] = {{{utterance.get(), three}, 2, three_two},
                               {{utterance.get(), three}, kVocab + 5, beyond}};
  ScorerAdvanceResult results[2];
  scorer->Advance({second, 2}, results);
  EXPECT_FLOAT_EQ(results[0].log_score, ReferenceLogProbs({3})[2]);
  EXPECT_FLOAT_EQ(results[1].log_score, ReferenceLogProbs({3})[1]);

  // The end of input scores the boundary, unless turned off.
  ScorerStateRef end = {utterance.get(), three_two};
  ScorerFinalScore final_score;
  scorer->Finish({&end, 1}, &final_score);
  EXPECT_FLOAT_EQ(final_score.log_score, ReferenceLogProbs({3, 2})[0]);
  EXPECT_TRUE(final_score.accepted);

  vocabulary.score_end = false;
  auto no_end = MakeRecurrentLmScorer(std::make_unique<FakeStep>(), vocabulary);
  auto no_end_utterance = no_end->NewUtterance();
  alignas(8) std::byte no_end_start[8];
  no_end->Start(no_end_utterance.get(), no_end_start);
  ScorerStateRef no_end_state = {no_end_utterance.get(), no_end_start};
  no_end->Finish({&no_end_state, 1}, &final_score);
  EXPECT_EQ(final_score.log_score, 0.0f);
  no_end->Info().destroy_state(no_end_start);

  for (std::byte *state : {start, three, three_two, beyond}) {
    info.destroy_state(state);
  }
  EXPECT_EQ(RecurrentLmLiveNodes(), 0);
}

// With max_batch 2, three nodes needed at once run in steps of 2 and 1,
// with the same scores as stepping each history alone.
TEST(RecurrentLmScorer, MaxBatchSplitsAWideStep) {
  Fixture f(2);
  alignas(8) std::byte start[8], children[3][8], grandchildren[3][8];
  f.scorer->Start(f.utterance.get(), start);
  UnitId units[3] = {'a', 'b', ' '};
  ScorerExtension wave[3];
  ScorerAdvanceResult results[3];
  for (int32_t i = 0; i != 3; ++i) {
    wave[i] = {{f.utterance.get(), start}, units[i], children[i]};
  }
  f.scorer->Advance({wave, 3}, results);
  f.step->batch_sizes.clear();
  for (int32_t i = 0; i != 3; ++i) {
    wave[i] = {{f.utterance.get(), children[i]}, 'b', grandchildren[i]};
  }
  f.scorer->Advance({wave, 3}, results);
  EXPECT_EQ(f.step->batch_sizes, std::vector<int32_t>({2, 1}));
  const int32_t model_ids[3] = {2, 3, 4};
  for (int32_t i = 0; i != 3; ++i) {
    EXPECT_FLOAT_EQ(results[i].log_score, ReferenceLogProbs({model_ids[i]})[3]);
  }
  for (int32_t i = 0; i != 3; ++i) {
    f.scorer->Info().destroy_state(grandchildren[i]);
    f.scorer->Info().destroy_state(children[i]);
  }
  f.scorer->Info().destroy_state(start);
  EXPECT_EQ(RecurrentLmLiveNodes(), 0);
}

TEST(RecurrentLmScorer, ComputesANodeOnlyWhenScoredAfterAndInOneBatch) {
  Fixture f;
  const ScorerInfo &info = f.scorer->Info();
  f.step->batch_sizes.clear();  // Drop the load-time step for the root.

  alignas(8) std::byte start[8], children[3][8], grandchildren[3][8];
  f.scorer->Start(f.utterance.get(), start);
  UnitId units[3] = {'a', 'b', ' '};
  ScorerExtension wave[3];
  ScorerAdvanceResult results[3];
  for (int32_t i = 0; i != 3; ++i) {
    wave[i] = {{f.utterance.get(), start}, units[i], children[i]};
  }
  f.scorer->Advance({wave, 3}, results);
  // The root was computed at load; extending it needs no model step.
  EXPECT_TRUE(f.step->batch_sizes.empty());

  // Scoring after two of the three children computes exactly those two,
  // together. The third is never computed.
  for (int32_t i = 0; i != 2; ++i) {
    wave[i] = {{f.utterance.get(), children[i]}, 'a', grandchildren[i]};
  }
  f.scorer->Advance({wave, 2}, results);
  EXPECT_EQ(f.step->batch_sizes, std::vector<int32_t>({2}));
  EXPECT_FLOAT_EQ(results[1].log_score, ReferenceLogProbs({3})[2]);

  // A copy shares the node; the node goes when the last reference does.
  int32_t before = RecurrentLmLiveNodes();
  info.copy_state(grandchildren[0], grandchildren[2]);
  EXPECT_EQ(RecurrentLmLiveNodes(), before);
  info.destroy_state(grandchildren[0]);
  EXPECT_EQ(RecurrentLmLiveNodes(), before);
  info.destroy_state(grandchildren[2]);
  EXPECT_EQ(RecurrentLmLiveNodes(), before - 1);

  // An uncomputed node keeps its parent alive after the parent's own state
  // is gone, because it still needs the parent's recurrent state.
  info.destroy_state(children[1]);
  EXPECT_EQ(RecurrentLmLiveNodes(), before - 1);
  info.destroy_state(grandchildren[1]);
  EXPECT_EQ(RecurrentLmLiveNodes(), before - 3);

  info.destroy_state(children[0]);
  info.destroy_state(children[2]);
  info.destroy_state(start);
  EXPECT_EQ(RecurrentLmLiveNodes(), 0);
}

TEST(RecurrentLmScorer, FusesIntoPrefixSearchAndReleasesItsNodes) {
  auto vocabulary = std::make_shared<CtcSearchVocabulary>();
  vocabulary->tokens.resize(4);
  vocabulary->tokens[1].starts_word = true;
  vocabulary->tokens[1].chars = {'a', 'b'};
  vocabulary->tokens[2].chars = {'b'};
  vocabulary->tokens[3].starts_word = true;
  vocabulary->tokens[3].chars = {'a'};

  // 30 frames cycling through favored tokens 1, blank, 2, 3.
  const int32_t num_frames = 30, vocab_size = 4;
  std::vector<float> log_probs(num_frames * vocab_size, std::log(0.1f));
  int32_t favored[4] = {1, 0, 2, 3};
  for (int32_t t = 0; t != num_frames; ++t) {
    log_probs[t * vocab_size + favored[t % 4]] = std::log(0.7f);
  }

  ASSERT_EQ(RecurrentLmLiveNodes(), 0);
  {
    Fixture f;
    f.step->batch_sizes.clear();
    CtcPrefixSearchOptions options;
    options.max_active = 4;
    options.max_token_candidates = 3;
    CtcPrefixSearch search(options, vocabulary, {f.scorer.get()}, {0.05f});
    search.Decode(log_probs.data(), num_frames);
    auto best = search.Finish(1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_FALSE(best[0].tokens.empty());

    // The reported LM score is the model's score of the output's characters.
    std::vector<int32_t> units;
    bool word_open = false;
    for (int32_t token : best[0].tokens) {
      const auto &info = vocabulary->tokens[token];
      if (info.starts_word && word_open) units.push_back(4);
      for (UnitId c : info.chars) units.push_back(c == 'a' ? 2 : 3);
      word_open = true;
    }
    double expected = 0;
    std::vector<int32_t> history;
    for (int32_t unit : units) {
      expected += ReferenceLogProbs(history)[unit];
      history.push_back(unit);
    }
    expected += ReferenceLogProbs(history)[0];
    EXPECT_NEAR(best[0].scorer_log_scores[0], expected, 1e-3);

    // Model steps are batched across the beam, not one per hypothesis.
    int32_t steps = static_cast<int32_t>(f.step->batch_sizes.size());
    int32_t items = 0;
    for (int32_t size : f.step->batch_sizes) items += size;
    EXPECT_GT(items, steps);
    // Nodes of pruned prefixes are freed as the search goes.
    EXPECT_LT(RecurrentLmLiveNodes(), items);
  }
  EXPECT_EQ(RecurrentLmLiveNodes(), 0);
}

}  // namespace
}  // namespace sherpa_onnx

// sherpa-onnx/csrc/recurrent-lm-scorer-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/recurrent-lm-scorer.h"

#include <cmath>
#include <memory>
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
  int32_t VocabSize() const override { return kVocab; }
  const std::vector<StateShape> &StateShapes() const override {
    return shapes_;
  }

  void Run(const std::vector<int64_t> &units,
           const std::vector<std::vector<float>> &states_in,
           const float **log_probs,
           std::vector<const float *> *states_out) override {
    size_t batch = units.size();
    batch_sizes.push_back(static_cast<int32_t>(batch));
    out_.resize(shapes_.size());
    for (size_t k = 0; k != shapes_.size(); ++k) {
      int32_t layers = shapes_[k].layers, hidden = shapes_[k].hidden;
      out_[k].resize(layers * batch * hidden);
      for (int32_t layer = 0; layer != layers; ++layer) {
        for (size_t i = 0; i != batch; ++i) {
          for (int32_t h = 0; h != hidden; ++h) {
            size_t at = (layer * batch + i) * hidden + h;
            out_[k][at] = 0.5f * states_in[k][at] +
                          (units[i] + 1) * 0.1f * (layer * hidden + h + 1);
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
    *log_probs = log_probs_.data();
    states_out->clear();
    for (const auto &state : out_) states_out->push_back(state.data());
  }

  std::vector<int32_t> batch_sizes;

 private:
  std::vector<StateShape> shapes_ = {{2, 3}, {1, 2}};
  std::vector<std::vector<float>> out_;
  std::vector<float> log_probs_;
};

// The same model stepped one history at a time: the log probability of each
// next unit after "units" (model IDs), starting from the boundary.
std::vector<float> ReferenceLogProbs(const std::vector<int32_t> &units) {
  FakeStep step;
  std::vector<std::vector<float>> state = {std::vector<float>(6, 0.0f),
                                           std::vector<float>(2, 0.0f)};
  std::vector<int32_t> history = {0};
  history.insert(history.end(), units.begin(), units.end());
  std::vector<float> result;
  for (int32_t unit : history) {
    const float *log_probs;
    std::vector<const float *> out;
    step.Run({unit}, state, &log_probs, &out);
    result.assign(log_probs, log_probs + kVocab);
    state[0].assign(out[0], out[0] + 6);
    state[1].assign(out[1], out[1] + 2);
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
  Fixture() {
    auto owned = std::make_unique<FakeStep>();
    step = owned.get();
    scorer = MakeRecurrentLmScorer(std::move(owned), Vocabulary());
    utterance = scorer->NewUtterance();
  }
  FakeStep *step;
  std::unique_ptr<Scorer> scorer;
  std::unique_ptr<ScorerUtterance> utterance;
};

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

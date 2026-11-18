// sherpa-onnx/csrc/transducer-prefix-search-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/transducer-prefix-search.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

constexpr float kLogZero = -std::numeric_limits<float>::infinity();
constexpr int32_t kVocab = 3;  // 0 blank, 1, 2
constexpr int32_t kMaxDuration = 2;

using Tokens = std::vector<int32_t>;

double LogAdd(double a, double b) {
  if (a == kLogZero) return b;
  if (b == kLogZero) return a;
  double high = std::max(a, b);
  return high + std::log1p(std::exp(std::min(a, b) - high));
}

// A stand-in for a transducer: log probabilities that depend on the frame
// and on the hypothesis's last token, as a real model's depend on the frame
// and the token history.
float Mix(int32_t a, int32_t b, int32_t c) {
  uint32_t h = static_cast<uint32_t>(a) * 2654435761u ^
               static_cast<uint32_t>(b + 7) * 40503u ^
               static_cast<uint32_t>(c + 3) * 2246822519u;
  h ^= h >> 13;
  h *= 1274126177u;
  return static_cast<float>(h % 1000) / 250.0f - 2.0f;
}

std::vector<float> TokenLogProbs(int32_t frame, int32_t last) {
  std::vector<float> row(kVocab);
  double total = kLogZero;
  for (int32_t v = 0; v != kVocab; ++v) {
    row[v] = Mix(frame, last, v);
    total = LogAdd(total, row[v]);
  }
  for (float &x : row) x -= total;
  return row;
}

std::vector<float> DurationLogProbs(int32_t frame, int32_t last) {
  std::vector<float> row(kMaxDuration + 1);
  double total = kLogZero;
  for (int32_t d = 0; d <= kMaxDuration; ++d) {
    row[d] = Mix(frame, last, 100 + d);
    total = LogAdd(total, row[d]);
  }
  for (float &x : row) x -= total;
  return row;
}

// The summed log probability of every output, by following every path of
// the topology through num_frames frames.
struct Enumerator {
  TransducerTopology topology;
  int32_t num_frames;
  int32_t max_symbols_per_frame;
  std::map<Tokens, double> outputs;

  void Visit(int32_t frame, int32_t round, Tokens *tokens, double log_prob) {
    if (frame >= num_frames) {
      auto inserted = outputs.emplace(*tokens, log_prob);
      if (!inserted.second) {
        inserted.first->second = LogAdd(inserted.first->second, log_prob);
      }
      return;
    }
    int32_t last = tokens->empty() ? -1 : tokens->back();
    std::vector<float> row = TokenLogProbs(frame, last);
    bool last_round = round + 1 >= max_symbols_per_frame;
    if (topology == TransducerTopology::kTdt) {
      std::vector<float> durations = DurationLogProbs(frame, last);
      for (int32_t d = 0; d <= kMaxDuration; ++d) {
        Visit(frame + std::max(d, 1), 0, tokens,
              log_prob + row[0] + durations[d]);
        for (int32_t v = 1; v != kVocab; ++v) {
          if (d == 0 && last_round) continue;
          tokens->push_back(v);
          Visit(frame + d, d == 0 ? round + 1 : 0, tokens,
                log_prob + row[v] + durations[d]);
          tokens->pop_back();
        }
      }
      return;
    }
    Visit(frame + 1, 0, tokens, log_prob + row[0]);
    bool stays = topology == TransducerTopology::kRnnt;
    if (stays && last_round) return;
    for (int32_t v = 1; v != kVocab; ++v) {
      tokens->push_back(v);
      Visit(stays ? frame : frame + 1, stays ? round + 1 : 0, tokens,
            log_prob + row[v]);
      tokens->pop_back();
    }
  }
};

std::shared_ptr<SearchVocabulary> Vocabulary() {
  auto vocabulary = std::make_shared<SearchVocabulary>();
  vocabulary->tokens.resize(kVocab);
  for (int32_t v = 1; v != kVocab; ++v) {
    vocabulary->tokens[v].chars = {'a' + v - 1};
  }
  return vocabulary;
}

// Runs the search over num_frames frames of the stand-in model.
void Decode(TransducerPrefixSearch *search,
            const std::vector<Scorer *> &scorers, TransducerTopology topology,
            int32_t num_frames) {
  std::vector<float> log_probs, durations;
  std::vector<ScorerExtension> extensions;
  std::vector<ScorerAdvanceResult> results;
  while (search->Frame() < num_frames) {
    do {
      log_probs.clear();
      durations.clear();
      for (uint32_t hypothesis : search->Frontier()) {
        int32_t last = search->LastTokenOf(hypothesis);
        auto row = TokenLogProbs(search->Frame(), last);
        log_probs.insert(log_probs.end(), row.begin(), row.end());
        auto duration_row = DurationLogProbs(search->Frame(), last);
        durations.insert(durations.end(), duration_row.begin(),
                         duration_row.end());
      }
      search->Expand(log_probs.data(), topology == TransducerTopology::kTdt
                                           ? durations.data()
                                           : nullptr);
      for (size_t s = 0; s != scorers.size(); ++s) {
        extensions.clear();
        while (search->GatherScorerWave(s, &extensions)) {
          results.assign(extensions.size(), ScorerAdvanceResult());
          scorers[s]->Advance({extensions.data(), extensions.size()},
                              results.data());
          search->ApplyScorerWave(s, results.data());
          extensions.clear();
        }
      }
    } while (search->EndRound());
    search->EndFrame();
  }
}

struct Case {
  TransducerTopology topology;
  int32_t num_frames;
  int32_t max_symbols_per_frame;
};

const Case kCases[] = {
    {TransducerTopology::kOneSymbolPerFrame, 6, 1},
    {TransducerTopology::kRnnt, 4, 3},
    {TransducerTopology::kTdt, 5, 2},
};

TransducerPrefixSearchOptions Exhaustive(const Case &c) {
  TransducerPrefixSearchOptions options;
  options.topology = c.topology;
  options.max_active = 100000;
  options.max_token_candidates = kVocab;
  options.max_symbols_per_frame = c.max_symbols_per_frame;
  options.max_duration = kMaxDuration;
  return options;
}

TEST(TransducerPrefixSearch, UnprunedSearchMatchesEnumeration) {
  for (const Case &c : kCases) {
    Enumerator enumerator = {
        c.topology, c.num_frames, c.max_symbols_per_frame, {}};
    Tokens tokens;
    enumerator.Visit(0, 0, &tokens, 0);
    double total = kLogZero;
    for (const auto &entry : enumerator.outputs) {
      total = LogAdd(total, entry.second);
    }
    // The stand-in model's transitions are normalized, so one symbol per
    // frame loses nothing; the other topologies drop what would exceed the
    // per-frame symbol limit.
    if (c.topology == TransducerTopology::kOneSymbolPerFrame) {
      EXPECT_NEAR(total, 0.0, 1e-4);
    } else {
      EXPECT_LT(total, 0.0);
    }

    TransducerPrefixSearch search(Exhaustive(c), Vocabulary(), {}, {});
    Decode(&search, {}, c.topology, c.num_frames);
    auto nbest = search.Finish(100000);
    ASSERT_EQ(nbest.size(), enumerator.outputs.size());
    for (const auto &hypothesis : nbest) {
      EXPECT_NEAR(hypothesis.model_log_prob,
                  enumerator.outputs.at(hypothesis.tokens), 2e-4);
    }
    auto best = std::max_element(
        enumerator.outputs.begin(), enumerator.outputs.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; });
    EXPECT_EQ(nbest[0].tokens, best->first);
    EXPECT_EQ(search.Best().tokens, best->first);
  }
}

TEST(TransducerPrefixSearch, ScorerFusionMatchesEnumeration) {
  BigramTableScorer scorer(ScorerUnit::kAsrToken, -1.0f);
  scorer.Set(BigramTableScorer::kStart, 2, -0.1f);
  scorer.Set(2, 1, -0.2f);
  scorer.Set(1, 1, -3.0f);
  scorer.SetFinal(1, -0.4f);
  const float weight = 0.8f, bonus = 0.3f;
  auto score_of = [&](const Tokens &tokens) {
    double score = bonus * tokens.size();
    UnitId previous = BigramTableScorer::kStart;
    for (int32_t token : tokens) {
      score += weight * scorer.LogScore(previous, token);
      previous = token;
    }
    return score + weight * scorer.FinalLogScore(previous);
  };

  for (const Case &c : kCases) {
    Enumerator enumerator = {
        c.topology, c.num_frames, c.max_symbols_per_frame, {}};
    Tokens tokens;
    enumerator.Visit(0, 0, &tokens, 0);
    std::pair<Tokens, double> expected = {{}, kLogZero};
    for (const auto &entry : enumerator.outputs) {
      double score = entry.second + score_of(entry.first);
      if (score > expected.second) expected = {entry.first, score};
    }

    TransducerPrefixSearchOptions options = Exhaustive(c);
    options.length_bonus = bonus;
    TransducerPrefixSearch search(options, Vocabulary(), {&scorer}, {weight});
    Decode(&search, {&scorer}, c.topology, c.num_frames);
    auto best = search.Finish(1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0].tokens, expected.first);
    EXPECT_NEAR(best[0].score, expected.second, 2e-4);
  }
}

// A second scorer with a negative weight subtracts its score (LODR), so
// its contribution raises scores; the search must not assume scorers only
// lower them.
TEST(TransducerPrefixSearch, NegativeWeightScorerMatchesEnumeration) {
  BigramTableScorer added(ScorerUnit::kAsrToken, -1.0f);
  added.Set(BigramTableScorer::kStart, 2, -0.1f);
  added.Set(2, 1, -0.2f);
  added.SetFinal(1, -0.4f);
  BigramTableScorer removed(ScorerUnit::kAsrToken, -1.5f);
  removed.Set(BigramTableScorer::kStart, 1, -0.3f);
  removed.Set(1, 1, -0.2f);
  removed.SetFinal(2, -0.1f);
  const float added_weight = 0.8f, removed_weight = -0.4f;
  auto score_of = [&](const Tokens &tokens) {
    double score = 0;
    UnitId previous = BigramTableScorer::kStart;
    for (int32_t token : tokens) {
      score += added_weight * added.LogScore(previous, token) +
               removed_weight * removed.LogScore(previous, token);
      previous = token;
    }
    return score + added_weight * added.FinalLogScore(previous) +
           removed_weight * removed.FinalLogScore(previous);
  };

  for (const Case &c : kCases) {
    Enumerator enumerator = {
        c.topology, c.num_frames, c.max_symbols_per_frame, {}};
    Tokens tokens;
    enumerator.Visit(0, 0, &tokens, 0);
    std::pair<Tokens, double> expected = {{}, kLogZero};
    for (const auto &entry : enumerator.outputs) {
      double score = entry.second + score_of(entry.first);
      if (score > expected.second) expected = {entry.first, score};
    }

    TransducerPrefixSearch search(Exhaustive(c), Vocabulary(),
                                  {&added, &removed},
                                  {added_weight, removed_weight});
    Decode(&search, {&added, &removed}, c.topology, c.num_frames);
    auto best = search.Finish(1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0].tokens, expected.first);
    EXPECT_NEAR(best[0].score, expected.second, 2e-4);
  }
}

// modified_beam_search as the existing decoders run it: each frame keeps the
// best steps before equal outputs are combined, one token counts as blank,
// and the result is the best score over length with the context counted.
TEST(TransducerPrefixSearch, PruneStepsReproducesModifiedBeamSearch) {
  const int32_t num_frames = 14, context_size = 2, unknown = 2;
  int32_t differs_from_default = 0;
  for (int32_t max_active : {1, 2, 4}) {
    for (int32_t shift = 0; shift != 12; ++shift) {
      // The reference: outputs with their log probabilities.
      std::map<Tokens, double> hyps = {{{}, 0.0}};
      for (int32_t t = 0; t != num_frames; ++t) {
        struct Step {
          double score;
          Tokens tokens;
        };
        std::vector<Step> steps;
        for (const auto &hyp : hyps) {
          int32_t last = hyp.first.empty() ? -1 : hyp.first.back();
          auto row = TokenLogProbs(t + 100 * shift, last);
          for (int32_t v = 0; v != kVocab; ++v) {
            Tokens tokens = hyp.first;
            if (v != 0 && v != unknown) tokens.push_back(v);
            steps.push_back({hyp.second + row[v], tokens});
          }
        }
        std::stable_sort(
            steps.begin(), steps.end(),
            [](const Step &a, const Step &b) { return a.score > b.score; });
        steps.resize(std::min<size_t>(steps.size(), max_active));
        hyps.clear();
        for (const Step &step : steps) {
          auto inserted = hyps.emplace(step.tokens, step.score);
          if (!inserted.second) {
            inserted.first->second = LogAdd(inserted.first->second, step.score);
          }
        }
      }
      Tokens expected;
      double expected_key = kLogZero;
      for (const auto &hyp : hyps) {
        double key = hyp.second / (context_size + hyp.first.size());
        if (key > expected_key) {
          expected_key = key;
          expected = hyp.first;
        }
      }

      TransducerPrefixSearchOptions options;
      options.topology = TransducerTopology::kOneSymbolPerFrame;
      options.max_active = max_active;
      options.max_token_candidates = kVocab;
      options.prune_steps = true;
      options.blank_like_token = unknown;
      options.rank_length_offset = context_size;
      auto vocabulary = Vocabulary();
      vocabulary->tokens[unknown].emit = false;

      // The stand-in model with this case's frames.
      auto decode = [&](TransducerPrefixSearch *search) {
        std::vector<float> log_probs;
        while (search->Frame() < num_frames) {
          log_probs.clear();
          for (uint32_t hypothesis : search->Frontier()) {
            auto row = TokenLogProbs(search->Frame() + 100 * shift,
                                     search->LastTokenOf(hypothesis));
            log_probs.insert(log_probs.end(), row.begin(), row.end());
          }
          search->Expand(log_probs.data(), nullptr);
          search->EndRound();
          search->EndFrame();
        }
      };
      TransducerPrefixSearch search(options, vocabulary, {}, {});
      decode(&search);
      auto best = search.Finish(1);
      ASSERT_EQ(best.size(), 1u);
      EXPECT_EQ(best[0].tokens, expected)
          << "max_active " << max_active << " shift " << shift;

      options.prune_steps = false;
      TransducerPrefixSearch plain(options, vocabulary, {}, {});
      decode(&plain);
      differs_from_default += plain.Finish(1)[0].tokens != expected;
    }
  }
  // The rule is a real difference from the default search.
  EXPECT_GT(differs_from_default, 0);
}

TEST(TransducerPrefixSearch, ZeroProbabilityTransitionsDoNotEnterQueues) {
  auto vocabulary = std::make_shared<SearchVocabulary>();
  vocabulary->tokens.resize(2);
  TransducerPrefixSearchOptions options;
  options.topology = TransducerTopology::kTdt;
  options.max_duration = 1;
  for (int32_t rounds : {1, 2}) {
    options.max_symbols_per_frame = rounds;
    TransducerPrefixSearch search(options, vocabulary, {}, {});
    const float tokens[] = {0, kLogZero};
    const float durations[] = {kLogZero, 0};
    for (int32_t frame = 0; frame != 4; ++frame) {
      search.Expand(tokens, durations);
      EXPECT_FALSE(search.EndRound());
      search.EndFrame();
      ASSERT_EQ(search.Frontier().size(), 1u);
      EXPECT_EQ(search.NumLiveRecords(), 1);
    }
    auto result = search.Finish();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].tokens.empty());
    EXPECT_FLOAT_EQ(result[0].score, 0);
  }
}

TEST(TransducerPrefixSearch, ZeroWeightIgnoresInfiniteScoresButNotRejection) {
  for (bool final_only : {false, true}) {
    BigramTableScorer scorer(ScorerUnit::kAsrToken, final_only ? 0 : kLogZero);
    scorer.SetFinal(1, kLogZero);
    scorer.Reject(BigramTableScorer::kStart, 2);
    TransducerPrefixSearch search(TransducerPrefixSearchOptions(), Vocabulary(),
                                  {&scorer}, {0});
    const float tokens[] = {-10, -1, 0};
    search.Expand(tokens, nullptr);
    RunScorerWaves(&search, {&scorer});
    search.EndRound();
    search.EndFrame();
    auto result = search.Finish();
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].tokens, Tokens({1}));
    EXPECT_FLOAT_EQ(result[0].score, -1);
    EXPECT_EQ(result[0].scorer_log_scores[0], kLogZero);
  }
}

TEST(TransducerPrefixSearch, PrunedSearchReleasesWrappedScorerStates) {
  ASSERT_EQ(HistoryScorer::LiveStates(), 0);
  for (const Case &c : kCases) {
    HistoryScorer scorer(ScorerUnit::kCharacter, -0.05f);
    TransducerPrefixSearchOptions options = Exhaustive(c);
    options.max_active = 3;
    options.max_token_candidates = 2;
    const int32_t num_frames = 40;
    TransducerPrefixSearch search(options, Vocabulary(), {&scorer}, {1.0f});
    Decode(&search, {&scorer}, c.topology, num_frames);
    auto best = search.Finish(3);
    ASSERT_FALSE(best.empty());
    EXPECT_EQ(best[0].tokens.size(), best[0].timestamps.size());
    for (size_t i = 1; i < best[0].timestamps.size(); ++i) {
      EXPECT_LE(best[0].timestamps[i - 1], best[0].timestamps[i]);
    }
    // What remains is the beam, what waits at later frames, and ancestors;
    // only the first two and the root keep scorer states.
    EXPECT_LE(search.NumLiveRecords(), 3 * 4 * (num_frames + 1));
    EXPECT_LE(HistoryScorer::LiveStates(), 3 * 4 + 1);
  }
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

// The survival trace changes no result, and a search pruned to exactly the
// limits it reports for the best result finds that result with that score.
TEST(TransducerPrefixSearch, SurvivalTraceLimitsReproduceTheResult) {
  for (const Case &c : kCases) {
    const int32_t num_frames = 40;
    auto run = [&](const TransducerPrefixSearchOptions &options,
                   TransducerWaitHistogram *waiting) {
      HistoryScorer scorer(ScorerUnit::kCharacter, -0.05f);
      TransducerPrefixSearch search(options, Vocabulary(), {&scorer}, {1.0f});
      Decode(&search, {&scorer}, c.topology, num_frames);
      auto best = search.Finish(1);
      if (waiting != nullptr) *waiting = search.WaitHistogram();
      return best.empty() ? TransducerPrefixHypothesis() : best[0];
    };
    TransducerPrefixSearchOptions options = Exhaustive(c);
    options.max_active = 3;
    options.max_token_candidates = 2;
    options.beam = 6;
    TransducerPrefixHypothesis plain = run(options, nullptr);

    options.trace_survival = true;
    TransducerWaitHistogram waiting;
    TransducerPrefixHypothesis traced = run(options, &waiting);
    ASSERT_FALSE(traced.tokens.empty());
    EXPECT_EQ(traced.tokens, plain.tokens);
    EXPECT_FLOAT_EQ(traced.score, plain.score);
    const TransducerSurvivalNeed &all = traced.required;
    const TransducerSurvivalNeed &dominant = traced.required_dominant;
    EXPECT_GE(all.rank, 1);
    EXPECT_LE(all.rank, options.max_active);
    EXPECT_LE(all.gap, options.beam);
    EXPECT_GE(all.wait_rank, 1);
    // The dominant path is one of the paths.
    EXPECT_GE(dominant.rank, 1);
    EXPECT_LE(dominant.rank, all.rank);
    EXPECT_LE(dominant.gap, all.gap);
    EXPECT_LE(dominant.wait_rank, all.wait_rank);
    EXPECT_LE(dominant.wait_gap, all.wait_gap);
    EXPECT_GT(waiting.rounds, 0);
    int64_t counted = 0;
    for (int64_t n : waiting.counts) counted += n;
    int64_t lengths = 0;
    for (int64_t n : waiting.waiting_lengths) lengths += n;
    EXPECT_EQ(counted, lengths);

    TransducerPrefixSearchOptions tight = options;
    tight.trace_survival = false;
    tight.max_active = all.rank;
    tight.beam = all.gap + 1e-3f;
    TransducerPrefixHypothesis again = run(tight, nullptr);
    EXPECT_EQ(again.tokens, traced.tokens);
    EXPECT_NEAR(again.score, traced.score, 1e-4);
  }
}

}  // namespace
}  // namespace sherpa_onnx

// sherpa-onnx/csrc/ctc-prefix-search-test.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/ctc-prefix-search.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/test-scorers.h"

namespace sherpa_onnx {
namespace {

constexpr float kLogZero = -std::numeric_limits<float>::infinity();

using Tokens = std::vector<int32_t>;

double LogAdd(double a, double b) {
  if (a == kLogZero) return b;
  if (b == kLogZero) return a;
  double high = std::max(a, b);
  return high + std::log1p(std::exp(std::min(a, b) - high));
}

// num_frames rows of log probabilities over vocab_size tokens, token 0 blank.
std::vector<float> RandomLogProbs(int32_t num_frames, int32_t vocab_size,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-3.0f, 3.0f);
  std::vector<float> log_probs(num_frames * vocab_size);
  for (int32_t t = 0; t != num_frames; ++t) {
    float *row = log_probs.data() + t * vocab_size;
    double total = kLogZero;
    for (int32_t v = 0; v != vocab_size; ++v) {
      row[v] = uniform(rng);
      total = LogAdd(total, row[v]);
    }
    for (int32_t v = 0; v != vocab_size; ++v) row[v] -= total;
  }
  return log_probs;
}

// The acoustic log probability of every output, by enumerating all
// alignments: summed over an output's alignments, or its best alignment.
std::map<Tokens, double> EnumerateOutputs(const std::vector<float> &log_probs,
                                          int32_t num_frames,
                                          int32_t vocab_size, bool sum) {
  std::map<Tokens, double> outputs;
  std::vector<int32_t> path(num_frames, 0);
  while (true) {
    double log_prob = 0;
    Tokens output;
    int32_t previous = 0;
    for (int32_t t = 0; t != num_frames; ++t) {
      log_prob += log_probs[t * vocab_size + path[t]];
      if (path[t] != 0 && path[t] != previous) output.push_back(path[t]);
      previous = path[t];
    }
    auto inserted = outputs.emplace(output, log_prob);
    if (!inserted.second) {
      double &total = inserted.first->second;
      total = sum ? LogAdd(total, log_prob) : std::max(total, log_prob);
    }
    int32_t t = 0;
    while (t != num_frames && ++path[t] == vocab_size) path[t++] = 0;
    if (t == num_frames) break;
  }
  return outputs;
}

std::shared_ptr<CtcSearchVocabulary> PlainVocabulary(int32_t vocab_size) {
  auto vocabulary = std::make_shared<CtcSearchVocabulary>();
  vocabulary->tokens.resize(vocab_size);
  for (int32_t v = 1; v < vocab_size; ++v) {
    vocabulary->tokens[v].chars = {'a' + v - 1};
  }
  return vocabulary;
}

// Options under which nothing is pruned.
CtcPrefixSearchOptions Exhaustive(int32_t vocab_size) {
  CtcPrefixSearchOptions options;
  options.max_active = 100000;
  options.max_token_candidates = vocab_size;
  return options;
}

// The best output and its score under "score_of", which adds everything but
// the acoustic log probability. Ties go to the smaller token sequence.
std::pair<Tokens, double> BestOf(
    const std::map<Tokens, double> &outputs,
    const std::function<double(const Tokens &)> &score_of) {
  std::pair<Tokens, double> best = {{}, kLogZero};
  for (const auto &entry : outputs) {
    double score = entry.second + score_of(entry.first);
    if (score > best.second) best = {entry.first, score};
  }
  return best;
}

TEST(CtcPrefixSearch, UnprunedSearchMatchesEnumerationWithoutScorers) {
  const int32_t num_frames = 5, vocab_size = 4;
  for (uint32_t seed = 0; seed != 20; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    for (bool sum : {true, false}) {
      auto outputs = EnumerateOutputs(log_probs, num_frames, vocab_size, sum);
      auto expected = BestOf(outputs, [](const Tokens &) { return 0.0; });

      CtcPrefixSearchOptions options = Exhaustive(vocab_size);
      options.alignment = sum ? CtcAlignmentMode::kSum : CtcAlignmentMode::kMax;
      CtcPrefixSearch search(options, PlainVocabulary(vocab_size), {}, {});
      search.Decode(log_probs.data(), num_frames);
      auto nbest = search.Finish(1000);

      // Sum mode keeps every output. Max mode with no scorer drops
      // hypotheses that another with the same last token beats.
      if (sum) {
        ASSERT_EQ(nbest.size(), outputs.size()) << "seed " << seed;
        EXPECT_EQ(search.NumRecombined(), 0);
        for (const auto &hypothesis : nbest) {
          EXPECT_NEAR(hypothesis.acoustic_log_prob,
                      outputs.at(hypothesis.tokens), 1e-4);
        }
      } else {
        ASSERT_FALSE(nbest.empty());
        EXPECT_LT(nbest.size(), outputs.size()) << "seed " << seed;
        EXPECT_GT(search.NumRecombined(), 0);
        // Only the best result is exact: a dropped hypothesis stops
        // feeding its descendants.
        EXPECT_NEAR(nbest[0].acoustic_log_prob, outputs.at(nbest[0].tokens),
                    1e-4);
      }
      EXPECT_EQ(nbest[0].tokens, expected.first) << "seed " << seed;
      EXPECT_NEAR(nbest[0].score, expected.second, 1e-4) << "seed " << seed;
    }
  }
}

TEST(CtcPrefixSearch, MaxModeRecombinesOnEqualScorerStatesExactly) {
  const int32_t num_frames = 6, vocab_size = 4;
  // A bigram scorer's state is the last token, so hypotheses ending in the
  // same token have the same future.
  BigramTableScorer bigram(ScorerUnit::kAsrToken, -2.0f);
  bigram.Set(BigramTableScorer::kStart, 2, -0.2f);
  bigram.Set(2, 3, -0.1f);
  bigram.Set(3, 1, -0.4f);
  bigram.SetFinal(1, -0.3f);
  const float weight = 0.7f, bonus = 0.3f;
  auto score_of = [&](const Tokens &tokens) {
    double score = bonus * tokens.size();
    UnitId previous = BigramTableScorer::kStart;
    for (int32_t token : tokens) {
      score += weight * bigram.LogScore(previous, token);
      previous = token;
    }
    return score + weight * bigram.FinalLogScore(previous);
  };

  for (uint32_t seed = 400; seed != 420; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    auto outputs = EnumerateOutputs(log_probs, num_frames, vocab_size, false);
    auto expected = BestOf(outputs, score_of);

    CtcPrefixSearchOptions options = Exhaustive(vocab_size);
    options.alignment = CtcAlignmentMode::kMax;
    options.length_bonus = bonus;
    CtcPrefixSearch search(options, PlainVocabulary(vocab_size), {&bigram},
                           {weight});
    search.Decode(log_probs.data(), num_frames);
    auto nbest = search.Finish(1000);
    ASSERT_FALSE(nbest.empty());
    EXPECT_EQ(nbest[0].tokens, expected.first) << "seed " << seed;
    EXPECT_NEAR(nbest[0].score, expected.second, 1e-4) << "seed " << seed;
    EXPECT_GT(search.NumRecombined(), 0);
    EXPECT_LT(nbest.size(), outputs.size());

    // A scorer whose state is the whole history never recombines.
    HistoryScorer history(ScorerUnit::kAsrToken, -0.1f);
    CtcPrefixSearch unmerged(options, PlainVocabulary(vocab_size), {&history},
                             {weight});
    unmerged.Decode(log_probs.data(), num_frames);
    EXPECT_EQ(unmerged.NumRecombined(), 0);
    EXPECT_EQ(unmerged.Finish(1000).size(), outputs.size());
  }
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

// Decodes with "scorer" and returns the best result.
CtcPrefixHypothesis BestWith(const CtcPrefixSearchOptions &options,
                             const std::vector<float> &log_probs,
                             int32_t vocab_size, Scorer *scorer,
                             int64_t *kept = nullptr) {
  CtcPrefixSearch search(options, PlainVocabulary(vocab_size), {scorer},
                         {0.8f});
  search.Decode(log_probs.data(), log_probs.size() / vocab_size);
  auto best = search.Finish(1);
  if (kept != nullptr) *kept = search.NumHypothesesKept();
  return best.empty() ? CtcPrefixHypothesis() : best[0];
}

TEST(CtcPrefixSearch, MaxModeTracedBeamsPlusEpsilonReproduceTheResult) {
  const int32_t num_frames = 16, vocab_size = 5;
  BigramTableScorer bigram(ScorerUnit::kAsrToken, -1.5f);
  bigram.Set(1, 2, -0.1f);
  bigram.Set(2, 3, -0.2f);
  HistoryScorer history(ScorerUnit::kAsrToken, -0.3f);
  // The bigram scorer recombines; the history scorer never does.
  Scorer *scorers[] = {&bigram, &history};
  int32_t narrower = 0;
  for (Scorer *scorer : scorers) {
    for (uint32_t seed = 500; seed != 540; ++seed) {
      auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
      CtcPrefixSearchOptions options;
      options.alignment = CtcAlignmentMode::kMax;
      options.max_active = 200;
      options.max_token_candidates = vocab_size;
      options.trace_survival = true;
      int64_t wide_kept = 0;
      auto wide = BestWith(options, log_probs, vocab_size, scorer, &wide_kept);
      ASSERT_GE(wide.required_score_beam, 0.0f);
      ASSERT_GE(wide.required_max_active, 1);

      // A score-gap beam just above the traced gap: same result and score.
      options.trace_survival = false;
      options.beam = wide.required_score_beam + 1e-3f;
      int64_t gap_kept = 0;
      auto by_gap = BestWith(options, log_probs, vocab_size, scorer, &gap_kept);
      EXPECT_EQ(by_gap.tokens, wide.tokens) << "seed " << seed;
      EXPECT_NEAR(by_gap.score, wide.score, 1e-5) << "seed " << seed;
      EXPECT_LE(gap_kept, wide_kept);
      narrower += gap_kept < wide_kept;

      // A hypothesis beam of exactly the traced rank: the same.
      options.beam = std::numeric_limits<float>::infinity();
      options.max_active = wide.required_max_active;
      auto by_rank = BestWith(options, log_probs, vocab_size, scorer);
      EXPECT_EQ(by_rank.tokens, wide.tokens) << "seed " << seed;
      EXPECT_NEAR(by_rank.score, wide.score, 1e-5) << "seed " << seed;
    }
  }
  // The traced beams are real restrictions, not the unlimited search again.
  EXPECT_GT(narrower, 40);
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

TEST(CtcPrefixSearch, SumModeTracedBeamsPlusEpsilonReproduceTheResult) {
  const int32_t num_frames = 14, vocab_size = 5;
  BigramTableScorer scorer(ScorerUnit::kAsrToken, -1.5f);
  scorer.Set(1, 2, -0.1f);
  scorer.Set(2, 3, -0.2f);
  for (uint32_t seed = 500; seed != 540; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    CtcPrefixSearchOptions options;
    options.max_active = 64;
    options.max_token_candidates = vocab_size;
    options.trace_survival = true;
    auto wide = BestWith(options, log_probs, vocab_size, &scorer);

    // Sum mode's trace covers every alignment that added to the output, so
    // the result and its score are reproduced, at a wider beam than the
    // output's main alignment alone would need.
    options.trace_survival = false;
    options.beam = wide.required_score_beam + 1e-3f;
    auto by_gap = BestWith(options, log_probs, vocab_size, &scorer);
    EXPECT_EQ(by_gap.tokens, wide.tokens) << "seed " << seed;
    EXPECT_NEAR(by_gap.score, wide.score, 1e-4) << "seed " << seed;
    EXPECT_EQ(by_gap.required_score_beam, 0.0f);

    options.beam = std::numeric_limits<float>::infinity();
    options.max_active = wide.required_max_active;
    auto by_rank = BestWith(options, log_probs, vocab_size, &scorer);
    EXPECT_EQ(by_rank.tokens, wide.tokens) << "seed " << seed;
    EXPECT_NEAR(by_rank.score, wide.score, 1e-4) << "seed " << seed;
  }
}

TEST(CtcPrefixSearch, ChunkBoundariesDoNotChangeTheResult) {
  const int32_t num_frames = 12, vocab_size = 5;
  auto log_probs = RandomLogProbs(num_frames, vocab_size, 7);
  BigramTableScorer scorer(ScorerUnit::kAsrToken, -1.5f);
  scorer.Set(1, 2, -0.1f);
  CtcPrefixSearchOptions options;
  options.max_active = 4;
  options.max_token_candidates = 3;

  CtcPrefixSearch whole(options, PlainVocabulary(vocab_size), {&scorer},
                        {0.8f});
  whole.Decode(log_probs.data(), num_frames);

  CtcPrefixSearch chunked(options, PlainVocabulary(vocab_size), {&scorer},
                          {0.8f});
  int32_t chunks[] = {1, 4, 2, 5};
  int32_t done = 0;
  for (int32_t chunk : chunks) {
    chunked.Decode(log_probs.data() + done * vocab_size, chunk);
    done += chunk;
  }
  ASSERT_EQ(done, num_frames);
  EXPECT_EQ(chunked.NumFramesDecoded(), num_frames);

  auto a = whole.Finish(4);
  auto b = chunked.Finish(4);
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i != a.size(); ++i) {
    EXPECT_EQ(a[i].tokens, b[i].tokens);
    EXPECT_EQ(a[i].timestamps, b[i].timestamps);
    EXPECT_FLOAT_EQ(a[i].score, b[i].score);
  }
  EXPECT_EQ(whole.Best().tokens, chunked.Best().tokens);
}

TEST(CtcPrefixSearch, TokenScorerFusionMatchesEnumeration) {
  const int32_t num_frames = 5, vocab_size = 4;
  BigramTableScorer scorer(ScorerUnit::kAsrToken, -2.0f);
  scorer.Set(BigramTableScorer::kStart, 2, -0.2f);
  scorer.Set(2, 3, -0.1f);
  scorer.Set(3, 3, -4.0f);
  scorer.SetFinal(3, -0.3f);
  scorer.SetFinal(1, -2.5f);
  const float weight = 0.7f, bonus = 0.4f;

  auto score_of = [&](const Tokens &tokens) {
    double score = bonus * tokens.size();
    UnitId previous = BigramTableScorer::kStart;
    for (int32_t token : tokens) {
      score += weight * scorer.LogScore(previous, token);
      previous = token;
    }
    return score + weight * scorer.FinalLogScore(previous);
  };

  for (uint32_t seed = 100; seed != 120; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    auto outputs = EnumerateOutputs(log_probs, num_frames, vocab_size, true);
    auto expected = BestOf(outputs, score_of);

    CtcPrefixSearchOptions options = Exhaustive(vocab_size);
    options.length_bonus = bonus;
    CtcPrefixSearch search(options, PlainVocabulary(vocab_size), {&scorer},
                           {weight});
    search.Decode(log_probs.data(), num_frames);
    auto best = search.Finish(1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0].tokens, expected.first) << "seed " << seed;
    EXPECT_NEAR(best[0].score, expected.second, 1e-4) << "seed " << seed;
    ASSERT_EQ(best[0].scorer_log_scores.size(), 1u);
    EXPECT_NEAR(best[0].score,
                best[0].acoustic_log_prob +
                    weight * best[0].scorer_log_scores[0] +
                    bonus * best[0].tokens.size(),
                1e-4);
  }
}

// Tokens: 1 "▁ab" (starts a word), 2 "c" (continues it), 3 "▁" alone.
std::shared_ptr<CtcSearchVocabulary> WordPieceVocabulary() {
  auto vocabulary = std::make_shared<CtcSearchVocabulary>();
  vocabulary->tokens.resize(4);
  vocabulary->tokens[1].starts_word = true;
  vocabulary->tokens[1].chars = {'a', 'b'};
  vocabulary->tokens[2].chars = {'c'};
  vocabulary->tokens[3].starts_word = true;
  return vocabulary;
}

// The characters a character scorer sees for "tokens", and the word count.
std::vector<UnitId> CharsOf(const CtcSearchVocabulary &vocabulary,
                            const Tokens &tokens, int32_t *num_words) {
  std::vector<UnitId> chars;
  bool word_open = false;
  *num_words = 0;
  for (int32_t token : tokens) {
    const auto &info = vocabulary.tokens[token];
    if (info.starts_word && word_open) {
      chars.push_back(' ');
      *num_words += 1;
      word_open = false;
    }
    chars.insert(chars.end(), info.chars.begin(), info.chars.end());
    word_open = word_open || !info.chars.empty();
  }
  *num_words += word_open ? 1 : 0;
  return chars;
}

TEST(CtcPrefixSearch, CharacterScorerSeesTextWithWordBoundaries) {
  auto vocabulary = WordPieceVocabulary();
  int32_t words = 0;
  EXPECT_EQ(CharsOf(*vocabulary, {1, 2, 3, 3, 1}, &words),
            std::vector<UnitId>({'a', 'b', 'c', ' ', 'a', 'b'}));
  EXPECT_EQ(words, 2);
  EXPECT_EQ(CharsOf(*vocabulary, {3, 2, 1}, &words),
            std::vector<UnitId>({'c', ' ', 'a', 'b'}));

  const int32_t num_frames = 5, vocab_size = 4;
  BigramTableScorer scorer(ScorerUnit::kCharacter, -1.0f);
  scorer.Set('a', 'b', -0.1f);
  scorer.Set('b', 'c', -0.2f);
  scorer.Set('c', ' ', -0.3f);
  scorer.Set(' ', 'a', -0.05f);
  scorer.Set('b', ' ', -2.5f);
  scorer.SetFinal('c', -0.2f);
  const float weight = 0.9f, bonus = 0.5f;

  auto score_of = [&](const Tokens &tokens) {
    int32_t num_words = 0;
    double score = 0;
    UnitId previous = BigramTableScorer::kStart;
    for (UnitId unit : CharsOf(*vocabulary, tokens, &num_words)) {
      score += weight * scorer.LogScore(previous, unit);
      previous = unit;
    }
    return score + weight * scorer.FinalLogScore(previous) + bonus * num_words;
  };

  for (uint32_t seed = 200; seed != 220; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    auto outputs = EnumerateOutputs(log_probs, num_frames, vocab_size, true);
    auto expected = BestOf(outputs, score_of);

    CtcPrefixSearchOptions options = Exhaustive(vocab_size);
    options.length_bonus = bonus;
    options.length_bonus_unit = LengthBonusUnit::kWord;
    CtcPrefixSearch search(options, vocabulary, {&scorer}, {weight});
    search.Decode(log_probs.data(), num_frames);
    auto best = search.Finish(1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(best[0].tokens, expected.first) << "seed " << seed;
    EXPECT_NEAR(best[0].score, expected.second, 1e-4) << "seed " << seed;
  }
}

TEST(CtcPrefixSearch, RejectedContinuationsNeverAppear) {
  const int32_t num_frames = 5, vocab_size = 4;
  BigramTableScorer scorer(ScorerUnit::kAsrToken, -1.0f);
  scorer.Reject(1, 2);
  scorer.Reject(BigramTableScorer::kStart, 3);

  auto allowed = [](const Tokens &tokens) {
    if (!tokens.empty() && tokens[0] == 3) return false;
    for (size_t i = 1; i < tokens.size(); ++i) {
      if (tokens[i - 1] == 1 && tokens[i] == 2) return false;
    }
    return true;
  };
  auto score_of = [&](const Tokens &tokens) {
    return allowed(tokens) ? -1.0 * tokens.size()
                           : static_cast<double>(kLogZero);
  };

  for (uint32_t seed = 300; seed != 310; ++seed) {
    auto log_probs = RandomLogProbs(num_frames, vocab_size, seed);
    auto outputs = EnumerateOutputs(log_probs, num_frames, vocab_size, true);
    auto expected = BestOf(outputs, score_of);

    CtcPrefixSearch search(Exhaustive(vocab_size), PlainVocabulary(vocab_size),
                           {&scorer}, {1.0f});
    search.Decode(log_probs.data(), num_frames);
    auto nbest = search.Finish(1000);
    ASSERT_FALSE(nbest.empty());
    EXPECT_EQ(nbest[0].tokens, expected.first) << "seed " << seed;
    EXPECT_NEAR(nbest[0].score, expected.second, 1e-4);
    for (const auto &hypothesis : nbest) {
      EXPECT_TRUE(allowed(hypothesis.tokens));
    }
  }
}

TEST(CtcPrefixSearch, WrappedScorerStatesAreAllReleased) {
  const int32_t num_frames = 60, vocab_size = 4;
  auto log_probs = RandomLogProbs(num_frames, vocab_size, 11);
  ASSERT_EQ(HistoryScorer::LiveStates(), 0);
  {
    // Character units over multi-character tokens use intermediate states.
    HistoryScorer scorer(ScorerUnit::kCharacter, -0.01f);
    CtcPrefixSearchOptions options;
    options.max_active = 3;
    options.max_token_candidates = 3;
    CtcPrefixSearch search(options, WordPieceVocabulary(), {&scorer}, {1.0f});
    search.Decode(log_probs.data(), num_frames);
    EXPECT_FALSE(search.Finish(3).empty());
    // Pruned prefixes without surviving descendants are reclaimed: what
    // remains is the beam and its ancestors.
    EXPECT_LE(search.NumLiveRecords(), 3 * (num_frames + 1));
    EXPECT_EQ(HistoryScorer::LiveStates(), search.NumLiveRecords());
  }
  EXPECT_EQ(HistoryScorer::LiveStates(), 0);
}

TEST(CtcPrefixSearch, TimestampsAreWhereEachTokenIsMostProbablyEmitted) {
  // Frames: token 1, blank, token 2, token 2.
  const float no = -20.0f, yes = -1e-4f;
  std::vector<float> log_probs = {no, yes, no,  yes, no, no,
                                  no, no,  yes, no,  no, yes};
  CtcPrefixSearch search(CtcPrefixSearchOptions(), PlainVocabulary(3), {}, {});
  search.Decode(log_probs.data(), 4);
  auto best = search.Finish(1);
  ASSERT_EQ(best.size(), 1u);
  EXPECT_EQ(best[0].tokens, Tokens({1, 2}));
  EXPECT_EQ(best[0].timestamps, Tokens({0, 2}));
  EXPECT_EQ(search.Best().tokens, Tokens({1, 2}));
}

TEST(CtcPrefixSearch, UtterancesCanShareOneScorerBatch) {
  const int32_t num_frames = 8, vocab_size = 4;
  HistoryScorer scorer(ScorerUnit::kCharacter, -0.2f);
  CtcPrefixSearchOptions options;
  options.max_active = 4;
  auto vocabulary = WordPieceVocabulary();
  auto first_frames = RandomLogProbs(num_frames, vocab_size, 21);
  auto second_frames = RandomLogProbs(num_frames, vocab_size, 22);

  std::vector<CtcPrefixHypothesis> separate[2];
  {
    CtcPrefixSearch first(options, vocabulary, {&scorer}, {1.0f});
    CtcPrefixSearch second(options, vocabulary, {&scorer}, {1.0f});
    first.Decode(first_frames.data(), num_frames);
    second.Decode(second_frames.data(), num_frames);
    separate[0] = first.Finish(4);
    separate[1] = second.Finish(4);
  }

  CtcPrefixSearch first(options, vocabulary, {&scorer}, {1.0f});
  CtcPrefixSearch second(options, vocabulary, {&scorer}, {1.0f});
  std::vector<ScorerExtension> extensions;
  std::vector<ScorerAdvanceResult> results;
  for (int32_t t = 0; t != num_frames; ++t) {
    first.BeginFrame(first_frames.data() + t * vocab_size);
    second.BeginFrame(second_frames.data() + t * vocab_size);
    while (true) {
      extensions.clear();
      bool first_has = first.GatherScorerWave(0, &extensions);
      size_t first_count = extensions.size();
      bool second_has = second.GatherScorerWave(0, &extensions);
      if (!first_has && !second_has) break;
      results.assign(extensions.size(), ScorerAdvanceResult());
      scorer.Advance({extensions.data(), extensions.size()}, results.data());
      if (first_has) first.ApplyScorerWave(0, results.data());
      if (second_has) second.ApplyScorerWave(0, results.data() + first_count);
    }
    first.EndFrame();
    second.EndFrame();
  }

  CtcPrefixSearch *searches[2] = {&first, &second};
  for (int32_t u = 0; u != 2; ++u) {
    auto batched = searches[u]->Finish(4);
    ASSERT_EQ(batched.size(), separate[u].size());
    for (size_t i = 0; i != batched.size(); ++i) {
      EXPECT_EQ(batched[i].tokens, separate[u][i].tokens);
      EXPECT_FLOAT_EQ(batched[i].score, separate[u][i].score);
    }
  }
}

}  // namespace
}  // namespace sherpa_onnx

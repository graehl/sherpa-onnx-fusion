// sherpa-onnx/csrc/test-scorers.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_TEST_SCORERS_H_
#define SHERPA_ONNX_CSRC_TEST_SCORERS_H_

// Small scorers for tests of the scorer interface and of searches that use
// it. Not built into the library.

#include <cstring>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/scorer.h"

namespace sherpa_onnx {

// A bigram table over units. Its state is the previous unit, so it
// recombines like an n-gram model with a one-unit context.
class BigramTableScorer : public Scorer {
 public:
  static constexpr UnitId kStart = -1;

  explicit BigramTableScorer(ScorerUnit unit, float default_log_score = -3.0f)
      : default_log_score_(default_log_score) {
    info_.unit = unit;
    info_.state_bytes = 8;
    info_.recombines = true;
  }

  // Log score of "unit" after "previous" (kStart at the start).
  void Set(UnitId previous, UnitId unit, float log_score) {
    bigram_[{previous, unit}] = log_score;
  }
  void SetFinal(UnitId previous, float log_score) {
    final_[previous] = log_score;
  }
  // Makes "unit" after "previous" a rejected continuation.
  void Reject(UnitId previous, UnitId unit) {
    rejected_[{previous, unit}] = true;
  }

  float LogScore(UnitId previous, UnitId unit) const {
    auto it = bigram_.find({previous, unit});
    return it == bigram_.end() ? default_log_score_ : it->second;
  }
  float FinalLogScore(UnitId previous) const {
    auto it = final_.find(previous);
    return it == final_.end() ? 0.0f : it->second;
  }

  const ScorerInfo &Info() const override { return info_; }
  std::unique_ptr<ScorerUtterance> NewUtterance() override { return nullptr; }

  void Start(ScorerUtterance * /*utterance*/, std::byte *state) override {
    Write(kStart, state);
  }

  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) override {
    for (size_t i = 0; i != extensions.size; ++i) {
      UnitId previous = Read(extensions[i].parent.bytes);
      UnitId unit = extensions[i].unit;
      if (rejected_.count({previous, unit}) != 0) {
        results[i].rejected = true;
        continue;
      }
      results[i].log_score = LogScore(previous, unit);
      results[i].rejected = false;
      Write(unit, extensions[i].child);
    }
  }

  void Finish(Span<const ScorerStateRef> states,
              ScorerFinalScore *results) override {
    for (size_t i = 0; i != states.size; ++i) {
      results[i].log_score = FinalLogScore(Read(states[i].bytes));
      results[i].accepted = true;
    }
  }

 private:
  static UnitId Read(const std::byte *state) {
    UnitId unit;
    std::memcpy(&unit, state, sizeof(unit));
    return unit;
  }
  static void Write(UnitId unit, std::byte *state) {
    std::memset(state, 0, 8);
    std::memcpy(state, &unit, sizeof(unit));
  }

  ScorerInfo info_;
  float default_log_score_;
  std::map<std::pair<UnitId, UnitId>, float> bigram_;
  std::map<UnitId, float> final_;
  std::map<std::pair<UnitId, UnitId>, bool> rejected_;
};

// A scorer whose state is a heap-allocated history behind a pointer in the
// state bytes, with per-utterance bookkeeping. Each unit scores
// log_score_per_unit times (1 + the history's length), so the score depends
// on the whole history. Counts live states to check copy and destroy.
class HistoryScorer : public Scorer {
 public:
  struct Utterance : public ScorerUtterance {
    int32_t advances = 0;
  };

  explicit HistoryScorer(ScorerUnit unit, float log_score_per_unit = -0.5f)
      : log_score_per_unit_(log_score_per_unit) {
    info_.unit = unit;
    info_.state_bytes = sizeof(History *);
    info_.copy_state = &Copy;
    info_.destroy_state = &Destroy;
  }

  static int32_t &LiveStates() {
    static int32_t live = 0;
    return live;
  }

  static const std::vector<UnitId> &HistoryOf(const std::byte *state) {
    return *Read(state);
  }

  const ScorerInfo &Info() const override { return info_; }
  std::unique_ptr<ScorerUtterance> NewUtterance() override {
    return std::make_unique<Utterance>();
  }

  void Start(ScorerUtterance * /*utterance*/, std::byte *state) override {
    Write(new History, state);
  }

  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) override {
    for (size_t i = 0; i != extensions.size; ++i) {
      const History &parent = *Read(extensions[i].parent.bytes);
      static_cast<Utterance *>(extensions[i].parent.utterance)->advances += 1;
      auto *child = new History(parent);
      child->push_back(extensions[i].unit);
      results[i].log_score = log_score_per_unit_ * child->size();
      results[i].rejected = false;
      Write(child, extensions[i].child);
    }
  }

  void Finish(Span<const ScorerStateRef> states,
              ScorerFinalScore *results) override {
    for (size_t i = 0; i != states.size; ++i) {
      results[i].log_score = 0;
      results[i].accepted = true;
    }
  }

 private:
  using History = std::vector<UnitId>;

  static History *Read(const std::byte *state) {
    History *history;
    std::memcpy(&history, state, sizeof(history));
    return history;
  }
  static void Write(History *history, std::byte *state) {
    std::memcpy(state, &history, sizeof(history));
    LiveStates() += 1;
  }
  static void Copy(const std::byte *from, std::byte *to) {
    Write(new History(*Read(from)), to);
  }
  static void Destroy(std::byte *state) {
    delete Read(state);
    LiveStates() -= 1;
  }

  ScorerInfo info_;
  float log_score_per_unit_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TEST_SCORERS_H_

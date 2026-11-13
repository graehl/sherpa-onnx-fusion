// sherpa-onnx/csrc/rescoring-scorer.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/rescoring-scorer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace sherpa_onnx {

namespace {

int32_t g_live_histories = 0;

// A hypothesis's units as a chain shared with the hypotheses it grew from.
struct Link {
  Link *parent;
  UnitId unit;
  int32_t length;  // Units up to and including this one.
  int32_t references;
};

Link *Read(const std::byte *state) {
  Link *link;
  std::memcpy(&link, state, sizeof(link));
  return link;
}

void Write(Link *link, std::byte *state) {
  std::memcpy(state, &link, sizeof(link));
}

void Release(Link *link) {
  while (link != nullptr && --link->references == 0) {
    Link *parent = link->parent;
    delete link;
    --g_live_histories;
    link = parent;
  }
}

void CopyState(const std::byte *from, std::byte *to) {
  Link *link = Read(from);
  if (link != nullptr) ++link->references;
  Write(link, to);
}

void DestroyState(std::byte *state) { Release(Read(state)); }

struct Utterance : public ScorerUtterance {
  std::unique_ptr<ScorerUtterance> wrapped;
};

class RescoringScorer : public Scorer {
 public:
  explicit RescoringScorer(std::unique_ptr<Scorer> scorer)
      : scorer_(std::move(scorer)) {
    info_.unit = scorer_->Info().unit;
    info_.state_bytes = sizeof(Link *);
    info_.copy_state = &CopyState;
    info_.destroy_state = &DestroyState;
    // Equal states are the same history, so nothing recombines on them.
    info_.recombines = false;
  }

  const ScorerInfo &Info() const override { return info_; }

  std::unique_ptr<ScorerUtterance> NewUtterance() override {
    auto utterance = std::make_unique<Utterance>();
    utterance->wrapped = scorer_->NewUtterance();
    return utterance;
  }

  void Start(ScorerUtterance * /*utterance*/, std::byte *state) override {
    Write(nullptr, state);
  }

  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) override {
    for (size_t i = 0; i != extensions.size; ++i) {
      Link *parent = Read(extensions[i].parent.bytes);
      if (parent != nullptr) ++parent->references;
      Write(new Link{parent, extensions[i].unit,
                     parent == nullptr ? 1 : parent->length + 1, 1},
            extensions[i].child);
      ++g_live_histories;
      results[i] = ScorerAdvanceResult();
    }
  }

  // Runs the wrapped scorer over every history, one unit per batched call
  // across the histories that are that long.
  void Finish(Span<const ScorerStateRef> states,
              ScorerFinalScore *results) override {
    const ScorerInfo &wrapped = scorer_->Info();
    size_t state_words = (wrapped.state_bytes + 7) / 8;
    size_t num = states.size;
    std::vector<std::vector<UnitId>> units(num);
    size_t longest = 0;
    for (size_t i = 0; i != num; ++i) {
      const Link *link = Read(states[i].bytes);
      units[i].resize(link == nullptr ? 0 : link->length);
      for (size_t k = units[i].size(); k != 0; link = link->parent) {
        units[i][--k] = link->unit;
      }
      longest = std::max(longest, units[i].size());
      results[i] = ScorerFinalScore();
    }

    // Each history's wrapped state alternates between two buffers.
    std::vector<uint64_t> buffers(num * 2 * state_words + 1);
    auto buffer = [&](size_t i, size_t which) {
      return reinterpret_cast<std::byte *>(buffers.data() +
                                           (i * 2 + which) * state_words);
    };
    auto wrapped_utterance = [&](size_t i) {
      return static_cast<Utterance *>(states[i].utterance)->wrapped.get();
    };
    for (size_t i = 0; i != num; ++i) {
      scorer_->Start(wrapped_utterance(i), buffer(i, 0));
    }

    std::vector<ScorerExtension> extensions;
    std::vector<size_t> which;
    std::vector<ScorerAdvanceResult> advanced;
    for (size_t step = 0; step != longest; ++step) {
      extensions.clear();
      which.clear();
      for (size_t i = 0; i != num; ++i) {
        if (!results[i].accepted || units[i].size() <= step) continue;
        extensions.push_back({{wrapped_utterance(i), buffer(i, step % 2)},
                              units[i][step],
                              buffer(i, (step + 1) % 2)});
        which.push_back(i);
      }
      advanced.assign(extensions.size(), ScorerAdvanceResult());
      scorer_->Advance({extensions.data(), extensions.size()}, advanced.data());
      for (size_t k = 0; k != which.size(); ++k) {
        size_t i = which[k];
        if (wrapped.destroy_state != nullptr) {
          wrapped.destroy_state(buffer(i, step % 2));
        }
        if (advanced[k].rejected) {
          // No state was written; this history has none to finish.
          results[i].accepted = false;
          units[i].clear();
          continue;
        }
        results[i].log_score += advanced[k].log_score;
      }
    }

    std::vector<ScorerStateRef> ends;
    which.clear();
    for (size_t i = 0; i != num; ++i) {
      if (!results[i].accepted) continue;
      ends.push_back({wrapped_utterance(i), buffer(i, units[i].size() % 2)});
      which.push_back(i);
    }
    std::vector<ScorerFinalScore> finals(ends.size());
    scorer_->Finish({ends.data(), ends.size()}, finals.data());
    for (size_t k = 0; k != which.size(); ++k) {
      size_t i = which[k];
      results[i].log_score += finals[k].log_score;
      results[i].accepted = finals[k].accepted;
      if (wrapped.destroy_state != nullptr) {
        wrapped.destroy_state(buffer(i, units[i].size() % 2));
      }
    }
  }

 private:
  std::unique_ptr<Scorer> scorer_;
  ScorerInfo info_;
};

}  // namespace

std::unique_ptr<Scorer> MakeRescoringScorer(std::unique_ptr<Scorer> scorer) {
  return std::make_unique<RescoringScorer>(std::move(scorer));
}

int32_t RescoringScorerLiveHistories() { return g_live_histories; }

}  // namespace sherpa_onnx

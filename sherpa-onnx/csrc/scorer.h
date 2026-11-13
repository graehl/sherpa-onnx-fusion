// sherpa-onnx/csrc/scorer.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_SCORER_H_
#define SHERPA_ONNX_CSRC_SCORER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/scorer-options.h"

// A scorer adds a log score to a search hypothesis as its output grows: a
// language model, or anything with the same state-and-advance contract.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// The symbol a scorer consumes one at a time.
enum class ScorerUnit {
  kAsrToken,   // a recognizer token ID
  kCharacter,  // a Unicode scalar value of the recognized text
};

// A recognizer token ID or a Unicode scalar value, by ScorerUnit.
using UnitId = int32_t;

// Maps a Unicode scalar value to a scorer's own index for that character:
// direct indexing, -1 for a character without an entry.
class CharacterIndex {
 public:
  int32_t Lookup(UnitId character) const {
    return static_cast<size_t>(character) < indices_.size()
               ? indices_[character]
               : -1;
  }

  // "character" and "index" must be nonnegative.
  void Set(UnitId character, int32_t index) {
    if (static_cast<size_t>(character) >= indices_.size()) {
      indices_.resize(static_cast<size_t>(character) + 1, -1);
    }
    indices_[character] = index;
  }

 private:
  // One entry per scalar up to the largest one set; gaps are -1.
  std::vector<int32_t> indices_;
};

// Resolves the "unit" option ("char" or "token"). Returns false otherwise.
bool ParseScorerUnit(const std::string &name, ScorerUnit *unit);

struct ScorerInfo {
  ScorerUnit unit = ScorerUnit::kCharacter;

  // A scorer's state for one hypothesis is this many bytes, 8-byte aligned,
  // in storage the decoder owns.
  size_t state_bytes = 0;

  // Optional. A null copy_state means memcpy and a null destroy_state means
  // nothing to release. A scorer whose real state is variable-length or
  // reference-counted keeps a pointer or index in the bytes and sets both.
  // copy_state writes an uninitialized buffer.
  void (*copy_state)(const std::byte *from, std::byte *to) = nullptr;
  void (*destroy_state)(std::byte *state) = nullptr;

  // True when equal state bytes imply equivalent futures, as for an n-gram
  // context. Requires canonical bytes: unused bytes zeroed.
  bool recombines = false;
};

// A scorer's own data for one utterance: caches and state storage that live
// until the decoder has discarded every state of that utterance.
class ScorerUtterance {
 public:
  virtual ~ScorerUtterance() = default;
};

// A borrowed contiguous range.
template <class T>
struct Span {
  T *data = nullptr;
  size_t size = 0;

  T *begin() const { return data; }
  T *end() const { return data + size; }
  T &operator[](size_t i) const { return data[i]; }
};

struct ScorerStateRef {
  ScorerUtterance *utterance;  // Null if the scorer's NewUtterance is null.
  const std::byte *bytes;
};

struct ScorerExtension {
  ScorerStateRef parent;
  UnitId unit;
  std::byte *child;  // Uninitialized; written unless the result is rejected.
};

struct ScorerAdvanceResult {
  float log_score = 0;    // Raw natural-log score of the unit.
  bool rejected = false;  // No such continuation; child left unwritten.
};

struct ScorerFinalScore {
  float log_score = 0;  // Raw end-of-input contribution.
  bool accepted = true;
};

// One coordinator calls a scorer serially; each call may mix utterances.
class Scorer {
 public:
  virtual ~Scorer() = default;

  virtual const ScorerInfo &Info() const = 0;

  // Per-utterance data, or null when the scorer needs none.
  virtual std::unique_ptr<ScorerUtterance> NewUtterance() = 0;

  // Writes the state before any unit into an uninitialized buffer.
  virtual void Start(ScorerUtterance *utterance, std::byte *state) = 0;

  // Scores one unit after each parent state and writes each child state.
  // "results" has extensions.size entries. A parent and its child may belong
  // to the same call only as separate extensions; no child buffer aliases a
  // parent in the same call.
  virtual void Advance(Span<const ScorerExtension> extensions,
                       ScorerAdvanceResult *results) = 0;

  // Scores the end of input after each state, leaving the states valid.
  // Called at the actual end of input, not at a streaming chunk boundary.
  virtual void Finish(Span<const ScorerStateRef> states,
                      ScorerFinalScore *results) = 0;
};

// A scorer that gains nothing from seeing a whole generation at once, such
// as an n-gram on the CPU: it scores one extension or one end of input at a
// time. Decoders still call Advance and Finish with every pending item.
class PerItemScorer : public Scorer {
 public:
  void Advance(Span<const ScorerExtension> extensions,
               ScorerAdvanceResult *results) final {
    for (size_t i = 0; i != extensions.size; ++i) {
      results[i] = AdvanceOne(extensions[i]);
    }
  }

  void Finish(Span<const ScorerStateRef> states,
              ScorerFinalScore *results) final {
    for (size_t i = 0; i != states.size; ++i) {
      results[i] = FinishOne(states[i]);
    }
  }

 protected:
  // The contract of Advance and Finish for a single item.
  virtual ScorerAdvanceResult AdvanceOne(const ScorerExtension &extension) = 0;
  virtual ScorerFinalScore FinishOne(const ScorerStateRef &state) = 0;
};

// Creates scorers of one registered type.
class ScorerFactory {
 public:
  virtual ~ScorerFactory() = default;

  // "extra" is the scorer entry's JSON object text of type-specific options.
  // Throws std::runtime_error for options or a model it cannot use.
  virtual std::unique_ptr<Scorer> Create(const CommonScorerOptions &common,
                                         const std::string &extra) const = 0;
};

// Registers a factory under ScorerOptions::type. Registration is serial and
// finishes before the first CreateScorer; a name registered twice throws.
void RegisterScorerFactory(const std::string &type,
                           std::unique_ptr<ScorerFactory> factory);

// Registers the scorer types built into this library, once: "onnx-rnn", and
// "kenlm" when built with SHERPA_ONNX_ENABLE_KENLM. CreateScorer calls it.
void RegisterBuiltinScorerFactories();

// Throws std::runtime_error naming the registered types for an unknown type.
std::unique_ptr<Scorer> CreateScorer(const ScorerOptions &options);

std::vector<std::unique_ptr<Scorer>> CreateScorers(
    const std::vector<ScorerOptions> &options);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SCORER_H_

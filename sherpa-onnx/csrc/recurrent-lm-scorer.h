// sherpa-onnx/csrc/recurrent-lm-scorer.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_RECURRENT_LM_SCORER_H_
#define SHERPA_ONNX_CSRC_RECURRENT_LM_SCORER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/scorer.h"

namespace sherpa_onnx {

// One step of a recurrent language model over a batch: from each item's
// previous unit and recurrent state to its next-unit log probabilities and
// new state. An LSTM has two state tensors and a GRU one.
class RecurrentLmStep {
 public:
  struct StateShape {
    int32_t layers = 0;
    int32_t hidden = 0;
  };

  virtual ~RecurrentLmStep() = default;

  virtual int32_t VocabSize() const = 0;
  virtual const std::vector<StateShape> &StateShapes() const = 0;

  // units[b] is each item's previous unit ID. states_in[k] holds state tensor
  // k as [layers, batch, hidden]. On return *log_probs is [batch, vocab] and
  // states_out[k] is [layers, batch, hidden]; both stay valid until the next
  // Run.
  virtual void Run(const std::vector<int64_t> &units,
                   const std::vector<std::vector<float>> &states_in,
                   const float **log_probs,
                   std::vector<const float *> *states_out) = 0;
};

struct RecurrentLmVocabulary {
  // The model's unit ID for each character it knows. The boundary between
  // words is the entry for ' '.
  CharacterIndex id_of_char;
  int32_t boundary_id = 0;  // Begins and ends a sentence.
  int32_t unknown_id = 1;   // Stands for a character without an entry.
};

// A character scorer over a recurrent LM. A hypothesis's state refers to a
// node holding the recurrent state after its last unit and, once computed,
// the next-unit distribution. A node is computed only when a unit is scored
// after it, in one batched step with the other nodes that call needs, and
// freed when no hypothesis or uncomputed successor refers to it.
std::unique_ptr<Scorer> MakeRecurrentLmScorer(
    std::unique_ptr<RecurrentLmStep> step, RecurrentLmVocabulary vocabulary);

// For tests: nodes currently allocated across all utterances of all
// recurrent LM scorers.
int32_t RecurrentLmLiveNodes();

// Scorer type "onnx-rnn": MakeRecurrentLmScorer over an ONNX model whose
// first input is the previous unit IDs, int64 [batch, 1], whose remaining
// inputs are recurrent state tensors [layers, batch, hidden], and whose
// outputs are log probabilities [batch, vocab] followed by the new states in
// the same order. The boundary between words is the unit
// CommonScorerOptions::space.
//
// Type-specific options ("extra"):
//   "vocab": file with one unit per line, in unit ID order. Required.
//   "vocab-encoding": "utf8" (default) or "hex" (each line the unit's UTF-8
//     bytes in hexadecimal).
//   "boundary-id", "unknown-id": defaults 0 and 1.
std::unique_ptr<ScorerFactory> MakeOnnxRecurrentLmScorerFactory();

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RECURRENT_LM_SCORER_H_

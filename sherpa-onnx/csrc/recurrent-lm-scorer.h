// sherpa-onnx/csrc/recurrent-lm-scorer.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_RECURRENT_LM_SCORER_H_
#define SHERPA_ONNX_CSRC_RECURRENT_LM_SCORER_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/scorer.h"

namespace sherpa_onnx {

// One step of a recurrent language model over a batch: from each item's
// previous unit and recurrent state to its next-unit log probabilities and
// new state. An LSTM has two state tensors and a GRU one.
//
// A step's recurrent states stay where its model runs, in device memory for a
// GPU, and are read only by later steps; the search needs, and the host
// receives, only the log probabilities.
class RecurrentLmStep {
 public:
  // The result of one Run for its batch.
  class Result {
   public:
    virtual ~Result() = default;
    // Item's log probabilities of every next unit, on the host; valid while
    // the result lives.
    virtual const float *LogProbs(int32_t item) const = 0;
  };

  // An item's state before a step: item `item` of an earlier result of the
  // same step, or the zero state when `result` is null.
  struct Parent {
    const Result *result = nullptr;
    int32_t item = 0;
  };

  virtual ~RecurrentLmStep() = default;

  virtual int32_t VocabSize() const = 0;

  // units[b] is each item's previous unit ID and parents[b] its state; the
  // parents' results stay alive for the call. May be called from several
  // threads at once.
  virtual std::shared_ptr<const Result> Run(
      const std::vector<int64_t> &units,
      const std::vector<Parent> &parents) const = 0;
};

struct RecurrentLmVocabulary {
  // The model's unit IDs are the recognizer's token IDs, for a model trained
  // on the recognizer's tokens; id_of_char is then unused.
  bool token_units = false;
  // With token_units: the model's unit for recognizer token t is
  // t + token_id_offset.
  int32_t token_id_offset = 0;
  // The end of input adds the score of boundary_id after the last unit, as
  // rescoring a whole sentence does. Shallow fusion in the existing
  // streaming decoders adds none.
  bool score_end = true;
  // Lower bound on each raw unit/EOS log score, before its fusion weight.
  float score_floor = -std::numeric_limits<float>::infinity();
  // The model's unit ID for each character it knows. The boundary between
  // words is the entry for ' '.
  CharacterIndex id_of_char;
  int32_t boundary_id = 0;  // Begins and ends a sentence.
  int32_t unknown_id = 1;   // Stands for a unit the model does not know.
};

// A character or recognizer-token scorer over a recurrent LM. A hypothesis's
// state refers to a node holding the recurrent state after its last unit and,
// once computed, the next-unit distribution. A node is computed only when a
// unit is scored after it, in one batched step with the other nodes that call
// needs, and freed when no hypothesis or uncomputed successor refers to it.
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
// CommonScorerOptions::space. With unit=token the model's unit IDs are the
// recognizer's token IDs and no vocab file is read; this is the form of the
// "with-state" RNN LMs that OnlineRnnLM loads.
//
// Type-specific options ("extra"):
//   "vocab": file with one unit per line, in unit ID order. Required for
//     unit=char.
//   "vocab-encoding": "utf8" (default) or "hex" (each line the unit's UTF-8
//     bytes in hexadecimal).
//   "boundary-id", "unknown-id": defaults 0 and 1 for unit=char; for
//     unit=token, 1 (the usual sos/eos ID) and the recognizer's blank, 0.
//   "id-offset": with unit=token, the model's unit for recognizer token t
//     is t + this. Default 0. Nothing checks that the two vocabularies agree.
//   "score-end": whether the end of input adds the boundary's score.
//     Default true.
//   "floor": nonpositive natural-log score floor per unit and end of input.
//     Omitted means no floor; recurrent state updates are unchanged.
std::unique_ptr<ScorerFactory> MakeOnnxRecurrentLmScorerFactory();

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RECURRENT_LM_SCORER_H_

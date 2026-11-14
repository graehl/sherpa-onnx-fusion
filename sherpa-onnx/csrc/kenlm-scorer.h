// sherpa-onnx/csrc/kenlm-scorer.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_KENLM_SCORER_H_
#define SHERPA_ONNX_CSRC_KENLM_SCORER_H_

#include <memory>

#include "sherpa-onnx/csrc/scorer.h"

namespace sherpa_onnx {

// Scorer type "kenlm": a KenLM n-gram model (ARPA or KenLM binary) over
// characters, each character one LM word. Built only with
// SHERPA_ONNX_ENABLE_KENLM.
// The boundary between words is the LM word CommonScorerOptions::space.
//
// With unit=token the model is over the recognizer's tokens instead, in one
// of two forms chosen by a type-specific option:
//   "tokens": path of the recognizer's token table. Each token is the LM
//     word spelled as its symbol there (for example "▁THE"); the words are
//     resolved once, when the scorer is created, and the table is not kept.
//   "id-offset": N. The model was built on token IDs: token t is the LM
//     word that is t + N in decimal (N = 0 when the IDs are the same).
//     Nothing checks that the model and the recognizer mean the same tokens
//     by those IDs.
// A token the model lacks is its <unk>.
//
// The state is KenLM's own n-gram state, so hypotheses that reach the same
// context, including by backing off, have equal state bytes.
std::unique_ptr<ScorerFactory> MakeKenLmScorerFactory();

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KENLM_SCORER_H_

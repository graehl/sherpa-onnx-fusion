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
// The boundary between words is the LM word CommonScorerOptions::space. It
// takes no type-specific options.
//
// The state is KenLM's own n-gram state, so hypotheses that reach the same
// context, including by backing off, have equal state bytes.
std::unique_ptr<ScorerFactory> MakeKenLmScorerFactory();

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_KENLM_SCORER_H_

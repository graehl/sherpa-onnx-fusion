// sherpa-onnx/csrc/rescoring-scorer.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_RESCORING_SCORER_H_
#define SHERPA_ONNX_CSRC_RESCORING_SCORER_H_

#include <memory>

#include "sherpa-onnx/csrc/scorer.h"

namespace sherpa_onnx {

// Wraps a scorer so that it does not take part in the search: every unit
// scores zero while a hypothesis grows, and the end of input scores the
// hypothesis's whole unit sequence with the wrapped scorer. A search then
// ranks its surviving hypotheses with the scorer and nothing else changes,
// which is what rescoring an n-best list does. Selected per scorer by
// ScorerOptions::rescore.
//
// For tests: RescoringScorerLiveHistories() counts history links currently
// allocated across all such scorers.
std::unique_ptr<Scorer> MakeRescoringScorer(std::unique_ptr<Scorer> scorer);

int32_t RescoringScorerLiveHistories();

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RESCORING_SCORER_H_

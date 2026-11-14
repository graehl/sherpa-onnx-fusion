// sherpa-onnx/csrc/fusion-config.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_FUSION_CONFIG_H_
#define SHERPA_ONNX_CSRC_FUSION_CONFIG_H_

#include <cstdint>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/scorer-options.h"

namespace sherpa_onnx {

// Scorers fused into a search, and how the search itself is run, for
// --decoding-method=prefix_beam_search.
// See topics/beam-search-and-scorer-fusion.md.
struct FusionConfig {
  // --scorer.N.<option>, or --scorer.<option> for the first.
  std::vector<ScorerOptions> scorers;

  // Added to a hypothesis's score per token or per word of its output.
  float length_bonus = 0;
  std::string length_bonus_unit = "token";  // "token" or "word"

  // CTC prefix search.
  std::string ctc_alignment = "sum";  // "sum" or "max"
  int32_t token_candidates = 8;
  // A hypothesis survives a frame only within this score of the best.
  // Zero or less means no limit. See the topic for the sweep behind 12.
  float score_beam = 12.0f;

  // The case of the text that character scorers are given: "keep" as the
  // recognizer writes it, or "lower". The recognizer's output is unchanged.
  std::string scorer_text_case = "keep";

  // Transducers: rank results by score divided by output length, counting
  // the model's leading context, as modified_beam_search does.
  bool length_normalize = false;
  // Stateless-decoder transducers: search as modified_beam_search does
  // (TransducerPrefixSearchOptions::prune_steps, the unknown-word token taken
  // as blank, and length_normalize), so that decoder's results are
  // reproduced, with or without scorers.
  bool modified_beam_search_rules = false;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FUSION_CONFIG_H_

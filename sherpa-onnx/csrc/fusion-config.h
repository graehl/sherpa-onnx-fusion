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
  int32_t ctc_token_candidates = 8;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_FUSION_CONFIG_H_

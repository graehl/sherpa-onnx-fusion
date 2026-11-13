// sherpa-onnx/csrc/scorer-options.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_SCORER_OPTIONS_H_
#define SHERPA_ONNX_CSRC_SCORER_OPTIONS_H_

#include <cstdint>
#include <string>

#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/provider-config.h"

namespace sherpa_onnx {

// Options every scorer type understands. Passed to the type's factory.
struct CommonScorerOptions {
  std::string unit = "char";  // "char" or "token"
  std::string model;
  // The scorer's own unit for the boundary between words (character units).
  std::string space = "▁";
  int32_t num_threads = 1;
  ProviderConfig provider;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

// One entry of a scorer list, set by --scorer.N.<option> or JSON; see
// topics/json-option-collections.md.
struct ScorerOptions {
  std::string type;  // Registered scorer type, e.g. "kenlm".
  CommonScorerOptions common;
  float weight = 1.0f;       // Default search weight of this scorer.
  std::string extra = "{}";  // JSON object text for the type's factory.

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SCORER_OPTIONS_H_

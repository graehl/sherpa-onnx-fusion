// sherpa-onnx/csrc/recognizer-options-json.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_RECOGNIZER_OPTIONS_JSON_H_
#define SHERPA_ONNX_CSRC_RECOGNIZER_OPTIONS_JSON_H_

#include <string>

#include "sherpa-onnx/csrc/offline-recognizer.h"

namespace sherpa_onnx {

// Applies a JSON object of command-line options to a recognizer config, as
// ApplyJsonOptions does for the options the config registers:
// {"scorer": [{"type": "kenlm", "model": "lm.bin"}], "length-bonus": 0.5}.
// It is how the language bindings set options their typed fields do not
// carry. Empty text changes nothing. On failure returns false and
// describes the problem in *error.
bool ApplyOptionsJson(const std::string &json_text,
                      OfflineRecognizerConfig *config, std::string *error);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_RECOGNIZER_OPTIONS_JSON_H_

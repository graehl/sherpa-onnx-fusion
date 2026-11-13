// sherpa-onnx/csrc/option-json.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OPTION_JSON_H_
#define SHERPA_ONNX_CSRC_OPTION_JSON_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// Applies a JSON object to registered options. Each scalar is assigned to
// the option whose name is its path: object members joined by ".", array
// positions as indices, so {"scorer": [{"weight": 0.3}]} assigns
// scorer.0.weight. An object or array at a path that is itself a string
// option is assigned as JSON text. On failure returns false and describes
// the problem and its path in *error; assignments before it remain applied.
// A value the option cannot convert ends the program, as on the command line.
// See topics/json-option-collections.md.
bool ApplyJsonOptions(const std::string &json_text, ParseOptions *po,
                      std::string *error);

// Registers --name=<file>: reads the file and applies it as ApplyJsonOptions
// does, at that option's position among the command-line options. A problem
// with the file ends the program.
void RegisterJsonOptionsFile(ParseOptions *po, const std::string &name,
                             const std::string &doc);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OPTION_JSON_H_

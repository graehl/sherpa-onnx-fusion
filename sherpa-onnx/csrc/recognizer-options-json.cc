// sherpa-onnx/csrc/recognizer-options-json.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/recognizer-options-json.h"

#include <string>

#include "sherpa-onnx/csrc/option-json.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

namespace {

template <typename Config>
bool ApplyToConfig(const std::string &json_text, Config *config,
                   std::string *error) {
  if (json_text.empty()) return true;
  ParseOptions po("");
  config->Register(&po);
  return ApplyJsonOptions(json_text, &po, error);
}

}  // namespace

bool ApplyOptionsJson(const std::string &json_text,
                      OfflineRecognizerConfig *config, std::string *error) {
  return ApplyToConfig(json_text, config, error);
}

}  // namespace sherpa_onnx

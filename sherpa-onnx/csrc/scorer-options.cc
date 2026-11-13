// sherpa-onnx/csrc/scorer-options.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/scorer-options.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void CommonScorerOptions::Register(ParseOptions *po) {
  po->Register("unit", &unit,
               "What the scorer consumes: char (characters of the recognized "
               "text) or token (recognizer tokens).");
  po->Register("model", &model, "Path to the scorer's model.");
  po->Register("space", &space,
               "The unit the scorer's model uses for the boundary between "
               "words, when it scores characters.");
  po->Register("num-threads", &num_threads,
               "Number of threads the scorer's model runs with.");
  provider.Register(po);
}

bool CommonScorerOptions::Validate() const {
  if (unit != "char" && unit != "token") {
    SHERPA_ONNX_LOGE("scorer unit must be char or token. Given: '%s'",
                     unit.c_str());
    return false;
  }
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("scorer num-threads must be positive. Given: %d",
                     num_threads);
    return false;
  }
  return provider.Validate();
}

std::string CommonScorerOptions::ToString() const {
  std::ostringstream os;
  os << "CommonScorerOptions(";
  os << "unit=\"" << unit << "\", ";
  os << "model=\"" << model << "\", ";
  os << "space=\"" << space << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "provider=" << provider.ToString() << ")";
  return os.str();
}

void ScorerOptions::Register(ParseOptions *po) {
  po->Register("type", &type, "Scorer type, for example kenlm or onnx-rnn.");
  common.Register(po);
  po->Register("weight", &weight,
               "Weight of this scorer's log score in the search.");
  po->Register("extra", &extra,
               "JSON object of options specific to the scorer type.");
}

bool ScorerOptions::Validate() const {
  if (type.empty()) {
    SHERPA_ONNX_LOGE("a scorer needs a type");
    return false;
  }
  return common.Validate();
}

std::string ScorerOptions::ToString() const {
  std::ostringstream os;
  os << "ScorerOptions(";
  os << "type=\"" << type << "\", ";
  os << "common=" << common.ToString() << ", ";
  os << "weight=" << weight << ", ";
  os << "extra=\"" << extra << "\")";
  return os.str();
}

}  // namespace sherpa_onnx

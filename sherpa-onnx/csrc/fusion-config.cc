// sherpa-onnx/csrc/fusion-config.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/fusion-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/option-collection.h"
#include "sherpa-onnx/csrc/option-json.h"

namespace sherpa_onnx {

void FusionConfig::Register(ParseOptions *po) {
  RegisterOptionCollection(po, "scorer", &scorers,
                           "A scorer (for example a language model) fused into "
                           "--decoding-method=prefix_beam_search.");
  RegisterJsonOptionsFile(
      po, "fusion-config",
      "JSON file of options, applied where it appears on the command line. "
      "A list of scorers is {\"scorer\": [{\"type\": ...}, ...]}.");
  po->Register("length-bonus", &length_bonus,
               "Used only when decoding_method is prefix_beam_search. Added "
               "to a hypothesis's score per --length-bonus-unit of output.");
  po->Register("length-bonus-unit", &length_bonus_unit,
               "What --length-bonus counts: token or word.");
  po->Register("ctc-alignment", &ctc_alignment,
               "Used only when decoding_method is prefix_beam_search. sum: "
               "score an output by the total probability of its alignments. "
               "max: by its best alignment.");
  po->Register("score-beam", &score_beam,
               "Used only when decoding_method is prefix_beam_search. A "
               "hypothesis survives a frame only if its score is within "
               "this much of the best hypothesis. 0 means no limit.");
  po->Register("token-candidates", &token_candidates,
               "Used only when decoding_method is prefix_beam_search. Number "
               "of most probable tokens per frame that may extend a "
               "hypothesis.");
}

bool FusionConfig::Validate() const {
  for (const auto &scorer : scorers) {
    if (!scorer.Validate()) return false;
  }
  if (length_bonus_unit != "token" && length_bonus_unit != "word") {
    SHERPA_ONNX_LOGE("--length-bonus-unit must be token or word. Given: '%s'",
                     length_bonus_unit.c_str());
    return false;
  }
  if (ctc_alignment != "sum" && ctc_alignment != "max") {
    SHERPA_ONNX_LOGE("--ctc-alignment must be sum or max. Given: '%s'",
                     ctc_alignment.c_str());
    return false;
  }
  if (token_candidates < 1) {
    SHERPA_ONNX_LOGE("--token-candidates must be positive. Given: %d",
                     token_candidates);
    return false;
  }
  return true;
}

std::string FusionConfig::ToString() const {
  std::ostringstream os;
  os << "FusionConfig(scorers=[";
  for (size_t i = 0; i != scorers.size(); ++i) {
    os << (i == 0 ? "" : ", ") << scorers[i].ToString();
  }
  os << "], ";
  os << "length_bonus=" << length_bonus << ", ";
  os << "length_bonus_unit=\"" << length_bonus_unit << "\", ";
  os << "ctc_alignment=\"" << ctc_alignment << "\", ";
  os << "token_candidates=" << token_candidates << ", ";
  os << "score_beam=" << score_beam << ")";
  return os.str();
}

}  // namespace sherpa_onnx

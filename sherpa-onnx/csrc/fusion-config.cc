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
  po->Register("ctc-prior", &ctc_prior,
               "Used only when decoding_method is prefix_beam_search, for "
               "CTC models. A text file with one 'token-id log-prior' pair "
               "per line. Each listed token other than blank has "
               "--ctc-prior-weight times its log prior subtracted from its "
               "log probability in every frame; unlisted tokens are "
               "unchanged.");
  po->Register("ctc-prior-weight", &ctc_prior_weight,
               "The weight of --ctc-prior. 0 turns it off.");
  po->Register("score-beam", &score_beam,
               "Used only when decoding_method is prefix_beam_search. A "
               "hypothesis survives a frame only if its score is within "
               "this much of the best hypothesis. 0 means no limit.");
  po->Register("scorer-text-case", &scorer_text_case,
               "Used only when decoding_method is prefix_beam_search. keep: "
               "character scorers are given the recognizer's text as "
               "written. lower: in lowercase, for a language model trained "
               "on lowercase text and a recognizer that writes capitals. The "
               "recognizer's output is unchanged.");
  po->Register("length-normalize", &length_normalize,
               "Used only when decoding_method is prefix_beam_search, for "
               "transducer models. Rank results by score divided by output "
               "length, as modified_beam_search does.");
  po->Register("modified-beam-search-rules", &modified_beam_search_rules,
               "Used only when decoding_method is prefix_beam_search, for "
               "transducer models with a stateless decoder. Search as "
               "modified_beam_search does: keep the best max-active-paths "
               "single steps per frame before combining equal outputs, take "
               "<unk> as blank, and rank results by score over length.");
  po->Register("token-candidates", &token_candidates,
               "Used only when decoding_method is prefix_beam_search. Number "
               "of most probable tokens per frame that may extend a "
               "hypothesis.");
  po->Register("trace-survival", &trace_survival,
               "Used only when decoding_method is prefix_beam_search, for "
               "offline transducer models. Measure the hypothesis beam and "
               "score beam each result needed, and how a cut of the lists "
               "waiting for later frames would have affected results and "
               "memory, and log a table of it at the end. Costs time.");
  po->Register("prune-waiting", &prune_waiting,
               "Used only when decoding_method is prefix_beam_search, for "
               "transducer models. At the end of every round, also cut each "
               "list of hypotheses waiting for a later frame to its best "
               "--waiting-max-active within --waiting-score-beam of its best, "
               "instead of only when that frame begins.");
  po->Register("waiting-max-active", &waiting_max_active,
               "With --prune-waiting: hypotheses each waiting list keeps; 0 "
               "for --max-active-paths.");
  po->Register("waiting-score-beam", &waiting_score_beam,
               "With --prune-waiting: a waiting hypothesis is kept only "
               "within this score of its list's best; 0 for --score-beam.");
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
  if (scorer_text_case != "keep" && scorer_text_case != "lower") {
    SHERPA_ONNX_LOGE("--scorer-text-case must be keep or lower. Given: '%s'",
                     scorer_text_case.c_str());
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
  os << "ctc_prior=\"" << ctc_prior << "\", ";
  os << "ctc_prior_weight=" << ctc_prior_weight << ", ";
  os << "scorer_text_case=\"" << scorer_text_case << "\", ";
  os << "length_normalize=" << (length_normalize ? "True" : "False") << ", ";
  os << "modified_beam_search_rules="
     << (modified_beam_search_rules ? "True" : "False") << ", ";
  os << "token_candidates=" << token_candidates << ", ";
  os << "score_beam=" << score_beam << ", ";
  os << "trace_survival=" << (trace_survival ? "True" : "False") << ", ";
  os << "prune_waiting=" << (prune_waiting ? "True" : "False") << ", ";
  os << "waiting_max_active=" << waiting_max_active << ", ";
  os << "waiting_score_beam=" << waiting_score_beam << ")";
  return os.str();
}

}  // namespace sherpa_onnx

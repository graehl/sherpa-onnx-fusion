// sherpa-onnx/python/csrc/fusion-config.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/python/csrc/fusion-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/scorer-options.h"

namespace sherpa_onnx {

static constexpr const char *kScorerConfigInitDoc = R"doc(
One scorer fused into ``prefix_beam_search``, e.g. a KenLM n-gram.

It binds the C++ ``ScorerOptions``; the options common to every scorer
type (model, unit, space, num_threads, provider) are flat attributes here,
as in the C API's ``SherpaOnnxScorerConfig``.

Args:
  type:
    Registered scorer type, e.g. ``kenlm``.
  model:
    Path to the scorer's model.
  unit:
    ``char`` or ``token``: what the scorer scores.
  space:
    The scorer's own unit for the boundary between words (character units).
  num_threads:
    Threads of the scorer's own model, if it runs one.
  provider:
    Provider of the scorer's own model, e.g. ``cpu``.
  weight:
    Search weight of the scorer.
  extra:
    JSON object text with options of the scorer type.
  rescore:
    Apply the scorer only to the hypotheses that survive the search, as
    n-best rescoring does, instead of during the search.
  stats:
    When the scorer is destroyed, log what it measured of its own work and
    memory (``onnx-rnn``: its widest model step and peak memory).
)doc";

static constexpr const char *kFusionConfigInitDoc = R"doc(
Scorers fused into a search, and how the search is run, for
``decoding_method="prefix_beam_search"``.

Args:
  scorers:
    A list of :class:`ScorerConfig`. Assign a whole list; appending to
    the attribute changes a copy.
  length_bonus:
    Added to a hypothesis's score per token or per word of its output.
  length_bonus_unit:
    ``token`` or ``word``.
  ctc_alignment:
    CTC prefix search: ``sum`` over alignments or ``max``.
  token_candidates:
    CTC prefix search: tokens considered per frame.
  score_beam:
    A hypothesis survives a frame only within this score of the best.
    Zero or less means no limit.
  scorer_text_case:
    The case of the text character scorers are given: ``keep`` or
    ``lower``. The recognizer's output is unchanged.
  length_normalize:
    Transducers: rank results by score divided by output length, as
    ``modified_beam_search`` does.
  modified_beam_search_rules:
    Stateless-decoder transducers: search as ``modified_beam_search``
    does, so that its results are reproduced, with or without scorers.
  ctc_prior:
    A label-prior correction: a file of ``token-id log-prior`` lines. Each
    listed token other than blank has ``ctc_prior_weight`` times its log
    prior subtracted from its log probability in every frame.
  ctc_prior_weight:
    The weight of ``ctc_prior``.
  trace_survival:
    Offline transducers: log at the end what pruning each result needed,
    and what cutting the lists of arrivals waiting for later frames would
    cost in results and save in entries. Costs time.
  prune_waiting:
    Transducers: at every round's end also cut each list of arrivals
    waiting for a later frame to its best ``waiting_max_active`` within
    ``waiting_score_beam`` of its best.
  waiting_max_active:
    Entries each waiting list keeps; 0 for twice the max active paths.
  waiting_score_beam:
    Score gap a waiting list keeps; 0 for half of ``score_beam``.
)doc";

static void PybindScorerConfig(py::module *m) {
  using PyClass = ScorerOptions;
  py::class_<PyClass>(*m, "ScorerConfig")
      .def(py::init([](const std::string &type, const std::string &model,
                       const std::string &unit, const std::string &space,
                       int32_t num_threads, const std::string &provider,
                       float weight, const std::string &extra, bool rescore,
                       bool stats) {
             PyClass o;
             o.type = type;
             o.common.model = model;
             o.common.unit = unit;
             o.common.space = space;
             o.common.num_threads = num_threads;
             o.common.provider.provider = provider;
             o.weight = weight;
             o.extra = extra;
             o.rescore = rescore;
             o.common.stats = stats;
             return o;
           }),
           py::arg("type") = "", py::arg("model") = "",
           py::arg("unit") = "char", py::arg("space") = "▁",
           py::arg("num_threads") = 1, py::arg("provider") = "cpu",
           py::arg("weight") = 1.0, py::arg("extra") = "{}",
           py::arg("rescore") = false, py::arg("stats") = true,
           kScorerConfigInitDoc)
      .def_readwrite("type", &PyClass::type)
      .def_property(
          "model", [](const PyClass &o) { return o.common.model; },
          [](PyClass &o, const std::string &v) { o.common.model = v; })
      .def_property(
          "unit", [](const PyClass &o) { return o.common.unit; },
          [](PyClass &o, const std::string &v) { o.common.unit = v; })
      .def_property(
          "space", [](const PyClass &o) { return o.common.space; },
          [](PyClass &o, const std::string &v) { o.common.space = v; })
      .def_property(
          "num_threads", [](const PyClass &o) { return o.common.num_threads; },
          [](PyClass &o, int32_t v) { o.common.num_threads = v; })
      .def_property(
          "provider",
          [](const PyClass &o) { return o.common.provider.provider; },
          [](PyClass &o, const std::string &v) {
            o.common.provider.provider = v;
          })
      .def_readwrite("weight", &PyClass::weight)
      .def_readwrite("extra", &PyClass::extra)
      .def_readwrite("rescore", &PyClass::rescore)
      .def_property(
          "stats", [](const PyClass &o) { return o.common.stats; },
          [](PyClass &o, bool v) { o.common.stats = v; })
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindFusionConfig(py::module *m) {
  PybindScorerConfig(m);

  using PyClass = FusionConfig;
  py::class_<PyClass>(*m, "FusionConfig")
      .def(py::init([](const std::vector<ScorerOptions> &scorers,
                       float length_bonus, const std::string &length_bonus_unit,
                       const std::string &ctc_alignment,
                       int32_t token_candidates, float score_beam,
                       const std::string &scorer_text_case,
                       bool length_normalize, bool modified_beam_search_rules,
                       const std::string &ctc_prior, float ctc_prior_weight,
                       bool trace_survival, bool prune_waiting,
                       int32_t waiting_max_active, float waiting_score_beam) {
             PyClass f;
             f.scorers = scorers;
             f.length_bonus = length_bonus;
             f.length_bonus_unit = length_bonus_unit;
             f.ctc_alignment = ctc_alignment;
             f.token_candidates = token_candidates;
             f.score_beam = score_beam;
             f.scorer_text_case = scorer_text_case;
             f.length_normalize = length_normalize;
             f.modified_beam_search_rules = modified_beam_search_rules;
             f.ctc_prior = ctc_prior;
             f.ctc_prior_weight = ctc_prior_weight;
             f.trace_survival = trace_survival;
             f.prune_waiting = prune_waiting;
             f.waiting_max_active = waiting_max_active;
             f.waiting_score_beam = waiting_score_beam;
             return f;
           }),
           py::arg("scorers") = std::vector<ScorerOptions>{},
           py::arg("length_bonus") = 0.0,
           py::arg("length_bonus_unit") = "token",
           py::arg("ctc_alignment") = "sum", py::arg("token_candidates") = 8,
           py::arg("score_beam") = 12.0, py::arg("scorer_text_case") = "keep",
           py::arg("length_normalize") = false,
           py::arg("modified_beam_search_rules") = false,
           py::arg("ctc_prior") = "", py::arg("ctc_prior_weight") = 1.0,
           py::arg("trace_survival") = false, py::arg("prune_waiting") = true,
           py::arg("waiting_max_active") = 0,
           py::arg("waiting_score_beam") = 0.0, kFusionConfigInitDoc)
      .def_readwrite("scorers", &PyClass::scorers)
      .def_readwrite("length_bonus", &PyClass::length_bonus)
      .def_readwrite("length_bonus_unit", &PyClass::length_bonus_unit)
      .def_readwrite("ctc_alignment", &PyClass::ctc_alignment)
      .def_readwrite("token_candidates", &PyClass::token_candidates)
      .def_readwrite("score_beam", &PyClass::score_beam)
      .def_readwrite("scorer_text_case", &PyClass::scorer_text_case)
      .def_readwrite("length_normalize", &PyClass::length_normalize)
      .def_readwrite("modified_beam_search_rules",
                     &PyClass::modified_beam_search_rules)
      .def_readwrite("ctc_prior", &PyClass::ctc_prior)
      .def_readwrite("ctc_prior_weight", &PyClass::ctc_prior_weight)
      .def_readwrite("trace_survival", &PyClass::trace_survival)
      .def_readwrite("prune_waiting", &PyClass::prune_waiting)
      .def_readwrite("waiting_max_active", &PyClass::waiting_max_active)
      .def_readwrite("waiting_score_beam", &PyClass::waiting_score_beam)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx

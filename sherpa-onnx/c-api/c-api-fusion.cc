// sherpa-onnx/c-api/c-api-fusion.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/c-api/c-api-fusion.h"

#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/recognizer-options-json.h"

namespace sherpa_onnx {

namespace {

void SetIfGiven(const char *value, std::string *out) {
  if (value && *value) *out = value;
}

// A zero is the default unless "literal" (a struct's literal_zeros) is set.
template <typename T>
void SetNumber(T value, bool literal, T *out) {
  if (literal || value != 0) *out = value;
}

// The same for a flag whose default is on: a zero means the default unless
// "literal" is set.
void SetFlag(int32_t value, bool literal, bool *out) {
  if (literal || value != 0) *out = value != 0;
}

ScorerOptions GetScorerOptions(const SherpaOnnxScorerConfig &c, bool literal) {
  ScorerOptions o;
  SetIfGiven(c.type, &o.type);
  SetIfGiven(c.model, &o.common.model);
  SetIfGiven(c.unit, &o.common.unit);
  SetIfGiven(c.space, &o.common.space);
  SetNumber(c.num_threads, literal, &o.common.num_threads);
  SetIfGiven(c.provider, &o.common.provider.provider);
  SetNumber(c.weight, literal, &o.weight);
  SetIfGiven(c.extra, &o.extra);
  o.rescore = c.rescore != 0;
  SetFlag(c.stats, literal, &o.common.stats);
  return o;
}

template <typename CConfig, typename Config>
bool ApplyJson(const CConfig *c, Config *config) {
  if (!c->options_json || !*c->options_json) return true;
  std::string error;
  if (!ApplyOptionsJson(c->options_json, config, &error)) {
    SHERPA_ONNX_LOGE("Error in options_json: %s", error.c_str());
    return false;
  }
  return true;
}

}  // namespace

FusionConfig GetFusionConfig(const SherpaOnnxFusionConfig &c) {
  FusionConfig f;
  bool literal = c.literal_zeros != 0;
  if (c.scorers) {
    for (int32_t i = 0; i < c.num_scorers; ++i) {
      f.scorers.push_back(GetScorerOptions(c.scorers[i], literal));
    }
  }
  f.length_bonus = c.length_bonus;
  SetIfGiven(c.length_bonus_unit, &f.length_bonus_unit);
  SetIfGiven(c.ctc_alignment, &f.ctc_alignment);
  SetNumber(c.token_candidates, literal, &f.token_candidates);
  SetNumber(c.score_beam, literal, &f.score_beam);
  SetIfGiven(c.scorer_text_case, &f.scorer_text_case);
  f.length_normalize = c.length_normalize != 0;
  f.modified_beam_search_rules = c.modified_beam_search_rules != 0;
  SetIfGiven(c.ctc_prior, &f.ctc_prior);
  SetNumber(c.ctc_prior_weight, literal, &f.ctc_prior_weight);
  f.trace_survival = c.trace_survival != 0;
  SetFlag(c.prune_waiting, literal, &f.prune_waiting);
  f.waiting_max_active = c.waiting_max_active;
  f.waiting_score_beam = c.waiting_score_beam;
  return f;
}

OfflineEncoderConfig GetOfflineEncoderConfig(
    const SherpaOnnxOfflineEncoderConfig &c) {
  OfflineEncoderConfig e;
  OfflineEncoderWindowConfig &w = e.window;
  bool literal = c.window.literal_zeros != 0;
  w.seconds = c.window.seconds;
  SetNumber(c.window.overlap, literal, &w.overlap);
  SetNumber(c.window.slack, literal, &w.slack);
  SetIfGiven(c.window.normalize, &w.normalize);
  SetIfGiven(c.window.weight, &w.weight);
  SetIfGiven(c.window.blend, &w.blend);
  SetNumber(c.window.center_power, literal, &w.center_power);
  w.context_frames = c.window.context_frames;
  w.context_fraction = c.window.context_fraction;
  SetIfGiven(c.lengths, &e.lengths);
  e.pad_to = c.pad_to;
  return e;
}

bool ApplyCOptionsJson(const SherpaOnnxOfflineRecognizerConfig *c,
                       OfflineRecognizerConfig *config) {
  return ApplyJson(c, config);
}

bool ApplyCOptionsJson(const SherpaOnnxOnlineRecognizerConfig *c,
                       OnlineRecognizerConfig *config) {
  return ApplyJson(c, config);
}

}  // namespace sherpa_onnx

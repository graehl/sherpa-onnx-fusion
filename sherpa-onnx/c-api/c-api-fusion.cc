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

ScorerOptions GetScorerOptions(const SherpaOnnxScorerConfig &c) {
  ScorerOptions o;
  SetIfGiven(c.type, &o.type);
  SetIfGiven(c.model, &o.common.model);
  SetIfGiven(c.unit, &o.common.unit);
  SetIfGiven(c.space, &o.common.space);
  if (c.num_threads > 0) o.common.num_threads = c.num_threads;
  SetIfGiven(c.provider, &o.common.provider.provider);
  if (c.weight != 0) o.weight = c.weight;
  SetIfGiven(c.extra, &o.extra);
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
  if (c.scorers) {
    for (int32_t i = 0; i < c.num_scorers; ++i) {
      f.scorers.push_back(GetScorerOptions(c.scorers[i]));
    }
  }
  f.length_bonus = c.length_bonus;
  SetIfGiven(c.length_bonus_unit, &f.length_bonus_unit);
  SetIfGiven(c.ctc_alignment, &f.ctc_alignment);
  if (c.token_candidates > 0) f.ctc_token_candidates = c.token_candidates;
  if (c.score_beam != 0) f.ctc_score_beam = c.score_beam;
  return f;
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

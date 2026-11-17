// sherpa-onnx/jni/fusion-config.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/jni/fusion-config.h"

#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/recognizer-options-json.h"

namespace sherpa_onnx {

namespace {

// As the SHERPA_ONNX_JNI_READ_* macros do for a field they cannot find.
void ReportMissingField(JNIEnv *env, const char *name) {
  SHERPA_ONNX_LOGE("Failed to get field ID for '%s'", name);
  if (env->ExceptionCheck()) {
    env->ExceptionDescribe();
    env->ExceptionClear();
  }
  jclass ex = env->FindClass("java/lang/RuntimeException");
  if (ex) {
    std::string msg = std::string("Failed to get field ID for ") + name;
    env->ThrowNew(ex, msg.c_str());
    env->DeleteLocalRef(ex);
  }
}

ScorerOptions GetScorerOptions(JNIEnv *env, jobject scorer, bool *ok) {
  ScorerOptions ans;
  *ok = false;
  jclass cls = env->GetObjectClass(scorer);

  SHERPA_ONNX_JNI_READ_STRING(ans.type, type, cls, scorer);
  SHERPA_ONNX_JNI_READ_STRING(ans.common.model, model, cls, scorer);
  SHERPA_ONNX_JNI_READ_STRING(ans.common.unit, unit, cls, scorer);
  SHERPA_ONNX_JNI_READ_STRING(ans.common.space, space, cls, scorer);
  SHERPA_ONNX_JNI_READ_INT(ans.common.num_threads, numThreads, cls, scorer);
  SHERPA_ONNX_JNI_READ_STRING(ans.common.provider.provider, provider, cls,
                              scorer);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.weight, weight, cls, scorer);
  SHERPA_ONNX_JNI_READ_STRING(ans.extra, extra, cls, scorer);
  SHERPA_ONNX_JNI_READ_BOOL(ans.rescore, rescore, cls, scorer);

  env->DeleteLocalRef(cls);
  *ok = true;
  return ans;
}

template <typename Config>
bool ApplyField(JNIEnv *env, jobject config, Config *ans) {
  jclass cls = env->GetObjectClass(config);
  jfieldID fid = env->GetFieldID(cls, "optionsJson", "Ljava/lang/String;");
  env->DeleteLocalRef(cls);
  if (fid == nullptr || env->ExceptionCheck()) {
    ReportMissingField(env, "optionsJson");
    return false;
  }

  auto s = static_cast<jstring>(env->GetObjectField(config, fid));
  std::string json = JStringToUtf8(env, s);
  if (s) env->DeleteLocalRef(s);
  if (json.empty()) return true;

  std::string error;
  if (!ApplyOptionsJson(json, ans, &error)) {
    SHERPA_ONNX_LOGE("Error in optionsJson: %s", error.c_str());
    return false;
  }
  return true;
}

}  // namespace

FusionConfig GetFusionConfig(JNIEnv *env, jobject fusion, bool *ok) {
  FusionConfig ans;
  *ok = true;
  if (fusion == nullptr) return ans;
  *ok = false;

  jclass cls = env->GetObjectClass(fusion);

  jfieldID fid =
      env->GetFieldID(cls, "scorers", "[Lcom/k2fsa/sherpa/onnx/ScorerConfig;");
  if (fid == nullptr || env->ExceptionCheck()) {
    ReportMissingField(env, "scorers");
    return ans;
  }
  auto scorers = static_cast<jobjectArray>(env->GetObjectField(fusion, fid));
  jsize n = scorers ? env->GetArrayLength(scorers) : 0;
  for (jsize i = 0; i != n; ++i) {
    jobject scorer = env->GetObjectArrayElement(scorers, i);
    if (scorer == nullptr) continue;
    bool scorer_ok = false;
    ans.scorers.push_back(GetScorerOptions(env, scorer, &scorer_ok));
    env->DeleteLocalRef(scorer);
    if (!scorer_ok) return ans;
  }
  if (scorers) env->DeleteLocalRef(scorers);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.length_bonus, lengthBonus, cls, fusion);
  SHERPA_ONNX_JNI_READ_STRING(ans.length_bonus_unit, lengthBonusUnit, cls,
                              fusion);
  SHERPA_ONNX_JNI_READ_STRING(ans.ctc_alignment, ctcAlignment, cls, fusion);
  SHERPA_ONNX_JNI_READ_INT(ans.token_candidates, tokenCandidates, cls, fusion);
  SHERPA_ONNX_JNI_READ_FLOAT(ans.score_beam, scoreBeam, cls, fusion);
  SHERPA_ONNX_JNI_READ_STRING(ans.scorer_text_case, scorerTextCase, cls,
                              fusion);

  env->DeleteLocalRef(cls);
  *ok = true;
  return ans;
}

bool ApplyOptionsJsonField(JNIEnv *env, jobject config,
                           OfflineRecognizerConfig *ans) {
  return ApplyField(env, config, ans);
}

bool ApplyOptionsJsonField(JNIEnv *env, jobject config,
                           OnlineRecognizerConfig *ans) {
  return ApplyField(env, config, ans);
}

}  // namespace sherpa_onnx

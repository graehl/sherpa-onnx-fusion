// sherpa-onnx/jni/fusion-config.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_JNI_FUSION_CONFIG_H_
#define SHERPA_ONNX_JNI_FUSION_CONFIG_H_

#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

// Reads a com.k2fsa.sherpa.onnx.FusionConfig. A null object gives the
// defaults. On a missing field, logs, raises a Java exception and sets *ok
// false.
FusionConfig GetFusionConfig(JNIEnv *env, jobject fusion, bool *ok);

// Applies the String field optionsJson of the Java recognizer config
// `config` to *ans, after every typed field has been read. Returns false,
// having logged the reason, if the field is missing or the JSON cannot be
// applied.
bool ApplyOptionsJsonField(JNIEnv *env, jobject config,
                           OfflineRecognizerConfig *ans);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_JNI_FUSION_CONFIG_H_

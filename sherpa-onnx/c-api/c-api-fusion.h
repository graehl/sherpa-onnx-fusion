// sherpa-onnx/c-api/c-api-fusion.h
//
// Copyright (c)  2026  graehl

// Conversions from the C API's structs for this fork's options to the core
// configs. Internal to the C API library; not installed.

#ifndef SHERPA_ONNX_C_API_C_API_FUSION_H_
#define SHERPA_ONNX_C_API_C_API_FUSION_H_

#include "sherpa-onnx/c-api/c-api.h"
#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/online-recognizer.h"

namespace sherpa_onnx {

// A zero or NULL field keeps the core default.
FusionConfig GetFusionConfig(const SherpaOnnxFusionConfig &c);
OfflineEncoderConfig GetOfflineEncoderConfig(
    const SherpaOnnxOfflineEncoderConfig &c);

// Applies c->options_json to *config. On failure logs the problem and
// returns false.
bool ApplyCOptionsJson(const SherpaOnnxOfflineRecognizerConfig *c,
                       OfflineRecognizerConfig *config);
bool ApplyCOptionsJson(const SherpaOnnxOnlineRecognizerConfig *c,
                       OnlineRecognizerConfig *config);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_C_API_C_API_FUSION_H_

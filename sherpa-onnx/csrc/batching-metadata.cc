// sherpa-onnx/csrc/batching-metadata.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/batching-metadata.h"

#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

bool BatchingSupported(const Ort::ModelMetadata &meta_data,
                       OrtAllocator *allocator, bool fallback) {
  const std::string value =
      LookupCustomModelMetaData(meta_data, "batching_supported", allocator);
  if (value.empty()) return fallback;
  if (value == "true") return true;
  if (value == "false") return false;
  SHERPA_ONNX_LOGE(
      "Model metadata batching_supported must be true or false, not '%s'",
      value.c_str());
  SHERPA_ONNX_EXIT(-1);
  return false;
}

}  // namespace sherpa_onnx

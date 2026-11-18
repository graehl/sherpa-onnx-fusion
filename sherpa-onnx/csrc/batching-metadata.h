// sherpa-onnx/csrc/batching-metadata.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_BATCHING_METADATA_H_
#define SHERPA_ONNX_CSRC_BATCHING_METADATA_H_

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

// Whether a model may decode several utterances in one padded call. The
// model's metadata "batching_supported", "true" or "false", decides when
// present: its exporter decoded a padded batch against each utterance alone
// and wrote what it found, which an input's presence cannot show (a graph
// can take lengths yet not mask every layer, or quantize with ranges taken
// over the whole batch). Absent, `fallback` decides, usually whether the
// graph takes lengths. Any other value exits with an error.
bool BatchingSupported(const Ort::ModelMetadata &meta_data,
                       OrtAllocator *allocator, bool fallback);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_BATCHING_METADATA_H_

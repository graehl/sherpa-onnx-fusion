// sherpa-onnx/csrc/encoder-output.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/encoder-output.h"

#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

EncoderOutputBatch::EncoderOutputBatch(Ort::Value values,
                                       const Ort::Value &lengths)
    : values_(std::move(values)) {
  auto shape = values_.GetTensorTypeAndShapeInfo().GetShape();
  max_frames_ = static_cast<int32_t>(shape[1]);
  dim_ = static_cast<int32_t>(shape[2]);
  data_ = values_.GetTensorData<float>();
  auto info = lengths.GetTensorTypeAndShapeInfo();
  auto n = static_cast<size_t>(info.GetShape()[0]);
  frames_.resize(n);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t *p = lengths.GetTensorData<int32_t>();
    frames_.assign(p, p + n);
  } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t *p = lengths.GetTensorData<int64_t>();
    for (size_t i = 0; i != n; ++i) frames_[i] = static_cast<int32_t>(p[i]);
  } else {
    SHERPA_ONNX_LOGE("Unsupported encoder output length type: %d",
                     static_cast<int32_t>(info.GetElementType()));
    SHERPA_ONNX_EXIT(-1);
  }
}

EncoderOutputBatch::EncoderOutputBatch(std::vector<float> rows, int32_t frames,
                                       int32_t dim)
    : rows_(std::move(rows)),
      data_(rows_.data()),
      max_frames_(frames),
      dim_(dim),
      frames_{frames} {}

}  // namespace sherpa_onnx

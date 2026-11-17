// sherpa-onnx/csrc/encoder-output.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_ENCODER_OUTPUT_H_
#define SHERPA_ONNX_CSRC_ENCODER_OUTPUT_H_

#include <cstdint>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace sherpa_onnx {

// The output of one encoder call, (N, T, D) floats on the host, and each
// item's number of valid frames. The streams whose rows it holds share it
// (OfflineStream::SetEncoderOutput), so keeping a model's output for a later
// search copies nothing; it is freed when the last of them is searched.
// For a CTC model the rows are log probabilities, for a transducer encoder
// frames.
class EncoderOutputBatch {
 public:
  // An encoder call's outputs: values (N, T, D) float, lengths (N) of int64
  // or int32.
  EncoderOutputBatch(Ort::Value values, const Ort::Value &lengths);
  // One item of `frames` rows of `dim` values, such as an utterance's
  // blended windows.
  EncoderOutputBatch(std::vector<float> rows, int32_t frames, int32_t dim);

  EncoderOutputBatch(const EncoderOutputBatch &) = delete;
  EncoderOutputBatch &operator=(const EncoderOutputBatch &) = delete;

  int32_t Size() const { return static_cast<int32_t>(frames_.size()); }
  int32_t MaxFrames() const { return max_frames_; }
  int32_t Dim() const { return dim_; }
  int32_t Frames(int32_t item) const { return frames_[item]; }
  // The item's first row; its rows are contiguous, and the next item's
  // start MaxFrames() rows later.
  const float *Row(int32_t item) const {
    return data_ + static_cast<size_t>(item) * max_frames_ * dim_;
  }

 private:
  Ort::Value values_{nullptr};
  std::vector<float> rows_;
  const float *data_ = nullptr;
  int32_t max_frames_ = 0;
  int32_t dim_ = 0;
  std::vector<int32_t> frames_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ENCODER_OUTPUT_H_

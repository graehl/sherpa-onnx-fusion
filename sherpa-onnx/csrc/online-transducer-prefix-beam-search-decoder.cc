// sherpa-onnx/csrc/online-transducer-prefix-beam-search-decoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/online-transducer-prefix-beam-search-decoder.h"

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/transducer-prefix-search.h"

namespace sherpa_onnx {

OnlineTransducerDecoderResult
OnlineTransducerPrefixBeamSearchDecoder::GetEmptyResult() const {
  // The same leading context the other decoders use: padding, then blank.
  OnlineTransducerDecoderResult r;
  r.tokens.assign(model_->ContextSize(), -1);
  r.tokens.back() = 0;
  return r;
}

void OnlineTransducerPrefixBeamSearchDecoder::StripLeadingBlanks(
    OnlineTransducerDecoderResult *r) const {
  int32_t context_size = model_->ContextSize();
  r->tokens.erase(r->tokens.begin(), r->tokens.begin() + context_size);
}

void OnlineTransducerPrefixBeamSearchDecoder::Decode(
    Ort::Value encoder_out, OnlineStream ** /*ss*/,
    std::vector<OnlineTransducerDecoderResult> *result) {
  Decode(std::move(encoder_out), result);
}

void OnlineTransducerPrefixBeamSearchDecoder::Decode(
    Ort::Value encoder_out,
    std::vector<OnlineTransducerDecoderResult> *result) {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t encoder_dim = static_cast<int32_t>(shape[2]);
  if (batch_size != static_cast<int32_t>(result->size())) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! encoder_out.size(0) %d, result.size(0): %d", batch_size,
        static_cast<int32_t>(result->size()));
    SHERPA_ONNX_EXIT(-1);
  }
  const float *p = encoder_out.GetTensorData<float>();
  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();
  OrtAllocator *allocator = model_->Allocator();

  for (int32_t b = 0; b != batch_size; ++b) {
    OnlineTransducerDecoderResult &r = (*result)[b];
    if (r.prefix_search == nullptr) {
      r.prefix_search = fusion_.NewSearch(options_);
      r.tokens.resize(context_size);  // An empty result: only its context.
      r.timestamps.clear();
    }
    TransducerPrefixSearch *search = r.prefix_search.get();
    // What precedes this segment for the model: the result's first tokens.
    std::vector<int64_t> before(r.tokens.begin(),
                                r.tokens.begin() + context_size);
    const float *frames = p + static_cast<size_t>(b) * num_frames * encoder_dim;
    int32_t first_frame = search->Frame();

    auto run_model = [&](int32_t frame, const std::vector<uint32_t> &frontier,
                         std::vector<float> *log_probs,
                         std::vector<float> * /*durations*/) {
      int64_t num = static_cast<int64_t>(frontier.size());
      std::array<int64_t, 2> context_shape = {num, context_size};
      Ort::Value decoder_input = Ort::Value::CreateTensor<int64_t>(
          allocator, context_shape.data(), context_shape.size());
      int64_t *context = decoder_input.GetTensorMutableData<int64_t>();
      for (int64_t i = 0; i != num; ++i) {
        int64_t *row = context + i * context_size;
        search->ContextOf(frontier[i], context_size, -1, row);
        int32_t missing = context_size - search->LengthOf(frontier[i]);
        for (int32_t k = 0; k < missing; ++k) {
          row[k] = before[context_size - missing + k];
        }
      }
      Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

      std::array<int64_t, 2> encoder_shape = {num, encoder_dim};
      Ort::Value encoder_frames = Ort::Value::CreateTensor<float>(
          allocator, encoder_shape.data(), encoder_shape.size());
      float *dst = encoder_frames.GetTensorMutableData<float>();
      const float *src =
          frames + static_cast<size_t>(frame - first_frame) * encoder_dim;
      for (int64_t i = 0; i != num; ++i) {
        std::copy_n(src, encoder_dim, dst + i * encoder_dim);
      }
      Ort::Value logit =
          model_->RunJoiner(std::move(encoder_frames), View(&decoder_out));
      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty_ > 0) {
        for (int64_t i = 0; i != num; ++i) {
          p_logit[i * vocab_size] -= blank_penalty_;
        }
      }
      LogSoftmax(p_logit, vocab_size, static_cast<int32_t>(num));
      log_probs->assign(p_logit, p_logit + num * vocab_size);
    };
    DecodeTransducerFrames(search, fusion_.Scorers(), first_frame + num_frames,
                           run_model);

    TransducerPrefixHypothesis best = search->Best();
    r.tokens = std::move(before);
    r.tokens.insert(r.tokens.end(), best.tokens.begin(), best.tokens.end());
    r.timestamps = std::move(best.timestamps);
    r.num_trailing_blanks = r.timestamps.empty()
                                ? search->Frame()
                                : search->Frame() - 1 - r.timestamps.back();
  }
}

}  // namespace sherpa_onnx

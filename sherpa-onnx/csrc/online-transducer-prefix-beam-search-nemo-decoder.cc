// sherpa-onnx/csrc/online-transducer-prefix-beam-search-nemo-decoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/online-transducer-prefix-beam-search-nemo-decoder.h"

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/keep-alive.h"
#include "sherpa-onnx/csrc/nemo-transducer-prefix-segment.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

using Decoder = OnlineTransducerPrefixBeamSearchNeMoDecoder;

Decoder::~OnlineTransducerPrefixBeamSearchNeMoDecoder() = default;

// cpplint takes this constructor's wrapped lines for namespace indentation.
// NOLINTBEGIN(whitespace/indent_namespace)
Decoder::OnlineTransducerPrefixBeamSearchNeMoDecoder(
    OnlineTransducerNeMoModel *model, const FusionConfig &config,
    int32_t max_active, const SymbolTable &symbol_table, float blank_penalty)
    : config_(StreamingFusionConfig(config)),
      max_active_(max_active),
      fusion_(std::make_shared<TransducerFusion>(
          config_, symbol_table, model->VocabSize() - 1, model->VocabSize())),
      blank_penalty_(blank_penalty),
      calls_(std::make_shared<NeMoTransducerCalls>()) {
  // NOLINTEND
  calls_->run_decoder = [model](Ort::Value tokens,
                                std::vector<Ort::Value> states) {
    return model->RunDecoder(std::move(tokens), std::move(states));
  };
  calls_->run_joiner = [model](Ort::Value encoder_out, Ort::Value decoder_out) {
    return model->RunJoiner(std::move(encoder_out), std::move(decoder_out));
  };
  calls_->init_states = [model]() { return model->GetDecoderInitStates(); };
  calls_->allocator = model->Allocator();
  calls_->vocab_size = model->VocabSize();
}

namespace {

void SetResult(TransducerPrefixHypothesis best, int32_t frames,
               OnlineTransducerDecoderResult *r) {
  r->tokens.assign(best.tokens.begin(), best.tokens.end());
  r->timestamps = std::move(best.timestamps);
  r->num_trailing_blanks =
      r->timestamps.empty() ? frames : frames - 1 - r->timestamps.back();
}

}  // namespace

void OnlineTransducerPrefixBeamSearchNeMoDecoder::Decode(Ort::Value encoder_out,
                                                         OnlineStream **ss,
                                                         int32_t n) const {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t encoder_dim = static_cast<int32_t>(shape[2]);
  const float *p = encoder_out.GetTensorData<float>();

  std::vector<TransducerPrefixSearch *> ending;
  std::vector<int32_t> ending_stream;
  for (int32_t b = 0; b != n; ++b) {
    OnlineTransducerDecoderResult &r = ss[b]->GetResult();
    if (r.prefix_segment == nullptr) {
      r.prefix_segment =
          KeepingAlive(std::make_unique<NeMoTransducerPrefixSegment>(
                           calls_.get(), fusion_.get(), config_, max_active_,
                           blank_penalty_),
                       fusion_, calls_);
    }
    r.prefix_segment->Decode(
        p + static_cast<size_t>(b) * num_frames * encoder_dim, num_frames,
        encoder_dim);
    TransducerPrefixSearch *search = r.prefix_segment->Search();
    if (search == nullptr) continue;
    SetResult(search->Best(), search->Frame(), &r);
    r.frame_offset += num_frames;
    int32_t ready = ss[b]->NumFramesReady();
    if (ready > 0 && ss[b]->IsLastFrame(ready - 1)) {
      ending.push_back(search);
      ending_stream.push_back(b);
    }
  }

  // Streams whose input has ended report the result with the scorers'
  // end-of-input scores, finished together.
  auto finished = TransducerPrefixSearch::FinishTogether(ending, 1);
  for (size_t i = 0; i != ending.size(); ++i) {
    SetResult(finished[i].empty() ? TransducerPrefixHypothesis{}
                                  : std::move(finished[i][0]),
              ending[i]->Frame(), &ss[ending_stream[i]]->GetResult());
  }
}

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/online-transducer-prefix-beam-search-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// --decoding-method=prefix_beam_search for streaming transducers with a
// stateless decoder: blank or one token per frame, as the streaming
// modified_beam_search does, fused with the configured scorers. With one
// token-unit scorer that is the existing shallow fusion.
//
// A segment's search lives in its result (prefix_search) and continues
// across chunks; the recognizer's reset at an endpoint makes an empty
// result, which starts a new search whose model context is the previous
// segment's last tokens, as for the other decoders. A result after each
// chunk is the search's best prefix so far; once a stream's input has
// ended, each decoded chunk's result includes the scorers' end-of-input
// scores, so the last one is the final result and a rescore-only scorer
// takes effect there. Streams of one call are searched one after another
// and finished together.
class OnlineTransducerPrefixBeamSearchDecoder : public OnlineTransducerDecoder {
 public:
  // Throws std::runtime_error if a scorer cannot be created.
  OnlineTransducerPrefixBeamSearchDecoder(OnlineTransducerModel *model,
                                          const FusionConfig &config,
                                          int32_t max_active,
                                          const SymbolTable &symbol_table,
                                          float blank_penalty)
      : model_(model),
        config_(StreamingFusionConfig(config)),
        options_(MakeTransducerPrefixSearchOptions(
            config_, max_active, TransducerTopology::kOneSymbolPerFrame,
            model->ContextSize())),
        fusion_(std::make_shared<TransducerFusion>(config_, symbol_table, 0,
                                                   model->VocabSize())),
        blank_penalty_(blank_penalty) {
    options_.blank_like_token = UnknownTokenAsBlank(config_, symbol_table);
  }

  OnlineTransducerDecoderResult GetEmptyResult() const override;

  void StripLeadingBlanks(OnlineTransducerDecoderResult *r) const override;

  void Decode(Ort::Value encoder_out,
              std::vector<OnlineTransducerDecoderResult> *result) override;

  void Decode(Ort::Value encoder_out, OnlineStream **ss,
              std::vector<OnlineTransducerDecoderResult> *result) override;

 private:
  OnlineTransducerModel *model_;
  FusionConfig config_;  // StreamingFusionConfig; declared before its users
  TransducerPrefixSearchOptions options_;
  // Shared with each stream's search, which may outlive the decoder.
  std::shared_ptr<TransducerFusion> fusion_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_

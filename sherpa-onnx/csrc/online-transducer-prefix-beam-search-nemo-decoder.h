// sherpa-onnx/csrc/online-transducer-prefix-beam-search-nemo-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_NEMO_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_NEMO_DECODER_H_

#include <memory>

#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-decoder.h"
#include "sherpa-onnx/csrc/online-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// --decoding-method=prefix_beam_search for streaming NeMo transducers
// (RNN-T), fused with the configured scorers. The search is
// NeMoTransducerPrefixSegment's, the same as offline; a segment's search
// lives in the stream's result (prefix_segment) and continues across
// chunks, and the recognizer's reset starts a new one. A result after each
// chunk is the search's best prefix so far; once the stream's input has
// ended it includes the scorers' end-of-input scores.
class OnlineTransducerPrefixBeamSearchNeMoDecoder
    : public OnlineTransducerNeMoDecoder {
 public:
  // Throws std::runtime_error if a scorer cannot be created.
  OnlineTransducerPrefixBeamSearchNeMoDecoder(OnlineTransducerNeMoModel *model,
                                              const FusionConfig &config,
                                              int32_t max_active,
                                              const SymbolTable &symbol_table,
                                              float blank_penalty);
  ~OnlineTransducerPrefixBeamSearchNeMoDecoder() override;

  void Decode(Ort::Value encoder_out, OnlineStream **ss,
              int32_t n) const override;

 private:
  FusionConfig config_;
  int32_t max_active_;
  TransducerFusion fusion_;
  float blank_penalty_;
  std::unique_ptr<NeMoTransducerCalls> calls_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_NEMO_DECODER_H_

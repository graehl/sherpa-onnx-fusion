// sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/scorer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transducer-prefix-search.h"

// --decoding-method=prefix_beam_search for offline transducer models:
// TransducerPrefixSearch fused with the configured scorers, with the
// recognizer's decoder and joiner run for the hypotheses it asks about.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// The search options a validated configuration, a hypothesis beam, and a
// model's topology describe.
TransducerPrefixSearchOptions MakeTransducerPrefixSearchOptions(
    const FusionConfig &config, int32_t max_active,
    TransducerTopology topology);

// Scorers, weights, and the search's view of the token table, shared by the
// decoders below.
class TransducerFusion {
 public:
  // Throws std::runtime_error if a scorer cannot be created.
  TransducerFusion(const FusionConfig &config, const SymbolTable &symbol_table,
                   int32_t blank_id, int32_t vocab_size);

  std::unique_ptr<TransducerPrefixSearch> NewSearch(
      const TransducerPrefixSearchOptions &options) const;
  const std::vector<Scorer *> &Scorers() const { return scorer_pointers_; }

 private:
  std::shared_ptr<SearchVocabulary> vocabulary_;
  std::vector<std::unique_ptr<Scorer>> scorers_;
  std::vector<Scorer *> scorer_pointers_;
  std::vector<float> weights_;
};

// For transducers with a stateless decoder (zipformer and other icefall
// models): blank or one token per frame, as modified_beam_search does. The
// decoder and joiner run once per frame for all of an utterance's
// hypotheses together.
class OfflineTransducerPrefixBeamSearchDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerPrefixBeamSearchDecoder(OfflineTransducerModel *model,
                                           const FusionConfig &config,
                                           int32_t max_active,
                                           const SymbolTable &symbol_table,
                                           float blank_penalty);

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out, Ort::Value encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  OfflineTransducerModel *model_;
  TransducerPrefixSearchOptions options_;
  TransducerFusion fusion_;
  float blank_penalty_;
};

// For NeMo transducers: RNN-T, or token-and-duration when the model is one.
// The prediction network is stateful; its output and state are kept per
// hypothesis and computed one hypothesis at a time.
class OfflineTransducerPrefixBeamSearchNeMoDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerPrefixBeamSearchNeMoDecoder(
      OfflineTransducerNeMoModel *model, const FusionConfig &config,
      int32_t max_active, const SymbolTable &symbol_table, float blank_penalty);

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out, Ort::Value encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  OfflineTransducerDecoderResult DecodeOne(const float *encoder_out,
                                           int32_t num_frames,
                                           int32_t encoder_dim) const;

  OfflineTransducerNeMoModel *model_;
  FusionConfig config_;
  int32_t max_active_;
  TransducerFusion fusion_;
  float blank_penalty_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_

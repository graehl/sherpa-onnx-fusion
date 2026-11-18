// sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_

#include <functional>
#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-transducer-decoder.h"
#include "sherpa-onnx/csrc/offline-transducer-model.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"
#include "sherpa-onnx/csrc/scorer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/transducer-prefix-search.h"
#include "sherpa-onnx/csrc/transducer-survival-summary.h"

// --decoding-method=prefix_beam_search for offline transducer models:
// TransducerPrefixSearch fused with the configured scorers, with the
// recognizer's decoder and joiner run for the hypotheses it asks about.
// See topics/beam-search-and-scorer-fusion.md.

namespace sherpa_onnx {

// The search options a validated configuration, a hypothesis beam, and a
// model's topology describe.
// "context_size" is how many leading entries modified_beam_search counts in
// a hypothesis's length before its first token (FusionConfig's
// length_normalize).
TransducerPrefixSearchOptions MakeTransducerPrefixSearchOptions(
    const FusionConfig &config, int32_t max_active, TransducerTopology topology,
    int32_t context_size);

// TransducerPrefixSearchOptions::blank_like_token for a configuration: the
// table's "<unk>" under modified_beam_search_rules, otherwise none.
int32_t UnknownTokenAsBlank(const FusionConfig &config,
                            const SymbolTable &symbol_table);

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

  // With FusionConfig::trace_survival: adds finished searches and their
  // results (FinishTogether's) to the summary logged when this is destroyed.
  void RecordSurvival(const std::vector<TransducerPrefixSearch *> &searches,
                      const std::vector<std::vector<TransducerPrefixHypothesis>>
                          &results) const;

 private:
  std::shared_ptr<SearchVocabulary> vocabulary_;
  std::vector<std::unique_ptr<Scorer>> scorers_;
  std::vector<Scorer *> scorer_pointers_;
  std::vector<float> weights_;
  std::unique_ptr<TransducerSurvivalSummary> survival_;  // Null if not traced.
};

// Fills the arrays TransducerPrefixSearch::Expand takes for the hypotheses
// "frontier" at "frame": log_probs, and duration_log_probs for kTdt only.
using TransducerRunModel = std::function<void(
    int32_t frame, const std::vector<uint32_t> &frontier,
    std::vector<float> *log_probs, std::vector<float> *duration_log_probs)>;

// Advances "search" until it stands at frame "end_frame", running the model
// and the scorers for each round.
void DecodeTransducerFrames(TransducerPrefixSearch *search,
                            const std::vector<Scorer *> &scorers,
                            int32_t end_frame,
                            const TransducerRunModel &run_model);

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

struct NeMoTransducerCalls;

// For NeMo transducers: RNN-T, or token-and-duration when the model is one.
// The search is NeMoTransducerPrefixSegment's, one per utterance.
class OfflineTransducerPrefixBeamSearchNeMoDecoder
    : public OfflineTransducerDecoder {
 public:
  OfflineTransducerPrefixBeamSearchNeMoDecoder(
      OfflineTransducerNeMoModel *model, const FusionConfig &config,
      int32_t max_active, const SymbolTable &symbol_table, float blank_penalty);
  ~OfflineTransducerPrefixBeamSearchNeMoDecoder() override;

  std::vector<OfflineTransducerDecoderResult> Decode(
      Ort::Value encoder_out, Ort::Value encoder_out_length,
      OfflineStream **ss = nullptr, int32_t n = 0) override;

 private:
  FusionConfig config_;
  int32_t max_active_;
  TransducerFusion fusion_;
  float blank_penalty_;
  std::unique_ptr<NeMoTransducerCalls> calls_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TRANSDUCER_PREFIX_BEAM_SEARCH_DECODER_H_

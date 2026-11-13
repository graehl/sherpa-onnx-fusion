// sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/ctc-prefix-search.h"
#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/offline-ctc-decoder.h"
#include "sherpa-onnx/csrc/scorer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// What a CTC prefix search needs to know about each recognizer token, from
// the recognizer's token table: its text as characters, whether it begins a
// word (its piece begins with a space), and whether it may be emitted
// (symbols such as <unk> and the blank may not).
//
// The rule is that a character scorer scores the text the recognizer would
// output for the tokens: a token's characters are those of
// SymbolTable::operator[], which the recognizer concatenates. Two things
// differ from that text on purpose: no boundary is scored before the first
// word or twice in a row, and the silence token "SIL" contributes nothing.
// Join markers that the recognizer does not itself remove, such as "##",
// "@@", or a "|" delimiter token, are therefore scored as written, and a
// table that has them is reported once in the log.
std::shared_ptr<CtcSearchVocabulary> MakeCtcSearchVocabulary(
    const SymbolTable &symbol_table, int32_t blank_id);

// The search options a validated configuration and a hypothesis beam describe.
CtcPrefixSearchOptions MakeCtcPrefixSearchOptions(const FusionConfig &config,
                                                  int32_t max_active);

// --decoding-method=prefix_beam_search for offline CTC models: CTC prefix
// beam search fused with the configured scorers. The utterances of a batch
// share each scorer call.
class OfflineCtcPrefixBeamSearchDecoder : public OfflineCtcDecoder {
 public:
  // Throws std::runtime_error if a scorer cannot be created.
  OfflineCtcPrefixBeamSearchDecoder(const FusionConfig &config,
                                    int32_t max_active,
                                    const SymbolTable &symbol_table,
                                    int32_t blank_id);

  std::vector<OfflineCtcDecoderResult> Decode(
      Ort::Value log_probs, Ort::Value log_probs_length) override;

 private:
  CtcPrefixSearchOptions options_;
  std::shared_ptr<CtcSearchVocabulary> vocabulary_;
  std::vector<std::unique_ptr<Scorer>> scorers_;
  std::vector<float> weights_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

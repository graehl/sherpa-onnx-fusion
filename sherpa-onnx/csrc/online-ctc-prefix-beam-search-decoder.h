// sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_ONLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_ONLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "sherpa-onnx/csrc/ctc-prefix-search.h"
#include "sherpa-onnx/csrc/fusion-config.h"
#include "sherpa-onnx/csrc/online-ctc-decoder.h"
#include "sherpa-onnx/csrc/scorer.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

// --decoding-method=prefix_beam_search for streaming CTC models. Each stream
// keeps its search across chunks (OnlineStream::GetCtcPrefixSearch), and the
// streams decoded in one call share each scorer call.
//
// A stream's result after each chunk is the search's best prefix so far,
// which later frames can revise. After OnlineStream::InputFinished(), each
// decoded chunk's result includes the scorers' end-of-input scores, so the
// last one is the final result. A stream whose input is declared finished
// with nothing left to decode keeps its last result unrevised.
// num_trailing_blanks is the number of frames since the best prefix's last
// token was most probably emitted.
class OnlineCtcPrefixBeamSearchDecoder : public OnlineCtcDecoder {
 public:
  // Throws std::runtime_error if a scorer cannot be created.
  OnlineCtcPrefixBeamSearchDecoder(const FusionConfig &config,
                                   int32_t max_active,
                                   const SymbolTable &symbol_table,
                                   int32_t blank_id);

  void Decode(const float *log_probs, int32_t batch_size, int32_t num_frames,
              int32_t vocab_size, std::vector<OnlineCtcDecoderResult> *results,
              OnlineStream **ss = nullptr, int32_t n = 0) override;

 private:
  CtcPrefixSearchOptions options_;
  std::mutex vocabulary_mutex_;
  std::shared_ptr<CtcSearchVocabulary> vocabulary_;
  std::vector<std::unique_ptr<Scorer>> scorers_;
  std::vector<Scorer *> scorer_pointers_;
  std::vector<float> weights_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_CTC_PREFIX_BEAM_SEARCH_DECODER_H_

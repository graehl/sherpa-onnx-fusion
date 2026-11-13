// sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/online-ctc-prefix-beam-search-decoder.h"

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/online-stream.h"

namespace sherpa_onnx {

OnlineCtcPrefixBeamSearchDecoder::OnlineCtcPrefixBeamSearchDecoder(
    const FusionConfig &config, int32_t max_active,
    const SymbolTable &symbol_table, int32_t blank_id)
    : options_(MakeCtcPrefixSearchOptions(config, max_active)),
      vocabulary_(MakeCtcSearchVocabulary(symbol_table, blank_id)),
      scorers_(CreateScorers(config.scorers)) {
  for (const auto &scorer : scorers_) scorer_pointers_.push_back(scorer.get());
  for (const auto &scorer : config.scorers) weights_.push_back(scorer.weight);
}

void OnlineCtcPrefixBeamSearchDecoder::Decode(
    const float *log_probs, int32_t batch_size, int32_t num_frames,
    int32_t vocab_size, std::vector<OnlineCtcDecoderResult> *results,
    OnlineStream **ss, int32_t n) {
  if (batch_size != static_cast<int32_t>(results->size()) || batch_size != n ||
      ss == nullptr) {
    SHERPA_ONNX_LOGE(
        "Size mismatch! log_probs.size(0) %d, results.size(0): %d, n: %d",
        batch_size, static_cast<int32_t>(results->size()), n);
    SHERPA_ONNX_EXIT(-1);
  }

  // A model may score more token IDs than the table names.
  if (static_cast<int32_t>(vocabulary_->tokens.size()) != vocab_size) {
    auto padded = std::make_shared<CtcSearchVocabulary>(*vocabulary_);
    CtcSearchToken absent;
    absent.emit = false;
    padded->tokens.resize(vocab_size, absent);
    vocabulary_ = std::move(padded);
  }

  std::vector<CtcPrefixSearch *> searches;
  for (int32_t b = 0; b != batch_size; ++b) {
    if (ss[b]->GetCtcPrefixSearch() == nullptr) {
      ss[b]->SetCtcPrefixSearch(std::make_shared<CtcPrefixSearch>(
          options_, vocabulary_, scorer_pointers_, weights_));
    }
    searches.push_back(ss[b]->GetCtcPrefixSearch());
  }

  DecodeFramesTogether(searches, scorer_pointers_, log_probs, num_frames,
                       vocab_size, nullptr);

  for (int32_t b = 0; b != batch_size; ++b) {
    OnlineCtcDecoderResult &r = (*results)[b];
    // Once the stream's input has ended, report the result with the
    // scorers' end-of-input scores, which can change which prefix is best.
    CtcPrefixHypothesis best = searches[b]->Best();
    int32_t ready = ss[b]->NumFramesReady();
    if (ready > 0 && ss[b]->IsLastFrame(ready - 1)) {
      auto final_best = searches[b]->Finish(1);
      if (!final_best.empty()) best = std::move(final_best[0]);
    }
    int32_t frames = searches[b]->NumFramesDecoded();
    r.tokens.assign(best.tokens.begin(), best.tokens.end());
    r.num_trailing_blanks =
        best.timestamps.empty() ? frames : frames - 1 - best.timestamps.back();
    r.timestamps = std::move(best.timestamps);
  }
}

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

std::shared_ptr<CtcSearchVocabulary> MakeCtcSearchVocabulary(
    const SymbolTable &symbol_table, int32_t blank_id, bool lowercase) {
  if (symbol_table.IsByteBpe()) {
    throw std::runtime_error(
        "prefix_beam_search does not support byte-BPE token tables yet");
  }
  auto vocabulary = std::make_shared<CtcSearchVocabulary>();
  int32_t num_join_markers = 0;
  vocabulary->blank_id = blank_id;
  vocabulary->tokens.resize(symbol_table.NumSymbols());
  for (int32_t id = 0; id != symbol_table.NumSymbols(); ++id) {
    CtcSearchToken &token = vocabulary->tokens[id];
    if (id == blank_id || !symbol_table.Contains(id)) {
      token.emit = false;
      continue;
    }
    // The table gives a piece's text with its leading word marker as a space.
    std::string text = symbol_table[id];
    if (text.size() > 2 && text.front() == '<' && text.back() == '>') {
      token.emit = false;
      continue;
    }
    if ((text.size() > 2 && text.compare(0, 2, "##") == 0) ||
        (text.size() > 2 && text.compare(text.size() - 2, 2, "@@") == 0) ||
        text == "|") {
      ++num_join_markers;
    }
    if (text == "SIL") {
      // A silence token: the recognizer leaves it out of its text.
      continue;
    }
    if (!text.empty() && text.front() == ' ') {
      token.starts_word = true;
      text.erase(0, 1);
    }
    if (lowercase) text = ToLowerCase(text);
    for (char32_t c : Utf8ToUtf32(text)) {
      token.chars.push_back(static_cast<UnitId>(c));
    }
  }
  if (num_join_markers != 0) {
    SHERPA_ONNX_LOGE(
        "%d tokens look like word-piece join markers (\"##x\", \"x@@\") or "
        "a \"|\" word delimiter. The recognizer writes them into its text "
        "unchanged, so a character scorer is given them as written; a "
        "language model trained on plain text will score them badly.",
        num_join_markers);
  }
  return vocabulary;
}

std::shared_ptr<const std::vector<float>> LoadCtcPriorOffsets(
    const std::string &filename, float weight) {
  std::ifstream in = OpenInputFile(filename);
  if (!in.is_open()) {
    throw std::runtime_error("--ctc-prior: cannot open '" + filename + "'");
  }
  auto offsets = std::make_shared<std::vector<float>>();
  std::string line;
  int32_t line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    std::istringstream fields(line);
    int32_t id = -1;
    float log_prior = 0;
    std::string rest;
    if (!(fields >> id >> log_prior) || (fields >> rest) || id < 0 ||
        !std::isfinite(log_prior) || log_prior > 0) {
      throw std::runtime_error(
          "--ctc-prior: '" + filename + "' line " +
          std::to_string(line_number) +
          " is not a token ID and a log prior (at most 0): " + line);
    }
    if (static_cast<size_t>(id) >= offsets->size()) offsets->resize(id + 1);
    (*offsets)[id] = -weight * log_prior;
  }
  return offsets;
}

CtcPrefixSearchOptions MakeCtcPrefixSearchOptions(const FusionConfig &config,
                                                  int32_t max_active) {
  CtcPrefixSearchOptions options;
  if (!config.ctc_prior.empty() && config.ctc_prior_weight != 0) {
    options.token_offsets =
        LoadCtcPriorOffsets(config.ctc_prior, config.ctc_prior_weight);
  }
  options.alignment = config.ctc_alignment == "max" ? CtcAlignmentMode::kMax
                                                    : CtcAlignmentMode::kSum;
  options.max_active = max_active;
  options.max_token_candidates = config.token_candidates;
  if (config.score_beam > 0) options.beam = config.score_beam;
  options.length_bonus = config.length_bonus;
  options.length_bonus_unit = config.length_bonus_unit == "word"
                                  ? LengthBonusUnit::kWord
                                  : LengthBonusUnit::kToken;
  return options;
}

OfflineCtcPrefixBeamSearchDecoder::OfflineCtcPrefixBeamSearchDecoder(
    const FusionConfig &config, int32_t max_active,
    const SymbolTable &symbol_table, int32_t blank_id)
    : options_(MakeCtcPrefixSearchOptions(config, max_active)),
      vocabulary_(MakeCtcSearchVocabulary(symbol_table, blank_id,
                                          ScorersWantLowercase(config))),
      scorers_(CreateScorers(config.scorers)) {
  for (const auto &scorer : config.scorers) weights_.push_back(scorer.weight);
  std::vector<Scorer *> scorers;
  for (const auto &scorer : scorers_) scorers.push_back(scorer.get());
  PrepareScorers(*vocabulary_, scorers, options_.word_boundary_char);
}

std::shared_ptr<CtcSearchVocabulary> VocabularyOfSize(
    int32_t vocab_size, std::mutex *mutex,
    std::shared_ptr<CtcSearchVocabulary> *vocabulary) {
  std::lock_guard<std::mutex> lock(*mutex);
  if (static_cast<int32_t>((*vocabulary)->tokens.size()) != vocab_size) {
    auto padded = std::make_shared<CtcSearchVocabulary>(**vocabulary);
    CtcSearchToken absent;
    absent.emit = false;
    padded->tokens.resize(vocab_size, absent);
    *vocabulary = std::move(padded);
  }
  return *vocabulary;
}

std::vector<OfflineCtcDecoderResult> OfflineCtcPrefixBeamSearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);
  const float *p_log_probs = log_probs.GetTensorData<float>();
  const int64_t *p_length = log_probs_length.GetTensorData<int64_t>();

  std::shared_ptr<CtcSearchVocabulary> vocabulary =
      VocabularyOfSize(vocab_size, &vocabulary_mutex_, &vocabulary_);

  std::vector<Scorer *> scorers;
  for (const auto &scorer : scorers_) scorers.push_back(scorer.get());

  std::vector<std::unique_ptr<CtcPrefixSearch>> searches;
  for (int32_t b = 0; b != batch_size; ++b) {
    searches.push_back(std::make_unique<CtcPrefixSearch>(options_, vocabulary,
                                                         scorers, weights_));
  }

  std::vector<CtcPrefixSearch *> search_pointers;
  for (const auto &search : searches) search_pointers.push_back(search.get());
  DecodeFramesTogether(search_pointers, scorers, p_log_probs, num_frames,
                       vocab_size, p_length);

  // The scorers' end-of-input calls serve the whole batch at once.
  auto finished = CtcPrefixSearch::FinishTogether(search_pointers, 1);
  std::vector<OfflineCtcDecoderResult> ans(batch_size);
  for (int32_t b = 0; b != batch_size; ++b) {
    auto &best = finished[b];
    if (best.empty()) continue;
    ans[b].tokens.assign(best[0].tokens.begin(), best[0].tokens.end());
    ans[b].timestamps = std::move(best[0].timestamps);
  }
  return ans;
}

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/kenlm-scorer.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/kenlm-scorer.h"

#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lm/model.hh"
#include "lm/state.hh"
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

constexpr float kLn10 = 2.302585092994046f;

class KenLmScorer : public PerItemScorer {
 public:
  // "tokens": the units are recognizer token IDs. With "token_words" not
  // empty, token_words[id] is each token's word in the model; with it empty,
  // the model's word for a token is "id_prefix" followed by its ID plus
  // "id_offset" in decimal.
  KenLmScorer(const std::string &model, const std::string &space, bool tokens,
              const std::vector<std::string> &token_words, int32_t id_offset,
              const std::string &id_prefix)
      : space_(space),
        ids_are_words_(tokens && token_words.empty()),
        id_offset_(id_offset),
        id_prefix_(id_prefix) {
    try {
      model_.reset(lm::ngram::LoadVirtual(model.c_str()));
    } catch (const std::exception &e) {
      throw std::runtime_error("kenlm scorer: cannot load '" + model +
                               "': " + e.what());
    }
    if (model_->StateSize() != sizeof(lm::ngram::State)) {
      throw std::runtime_error("kenlm scorer: unsupported model state in '" +
                               model + "'");
    }
    info_.unit = tokens ? ScorerUnit::kAsrToken : ScorerUnit::kCharacter;
    info_.state_bytes = sizeof(lm::ngram::State);
    info_.recombines = true;
    end_sentence_ = model_->BaseVocabulary().EndSentence();
    for (const std::string &word : token_words) {
      // A token the table does not name, or the model lacks, is its <unk>.
      word_of_token_.push_back(model_->BaseVocabulary().Index(word));
    }
  }

  const ScorerInfo &Info() const override { return info_; }
  std::unique_ptr<ScorerUtterance> NewUtterance() override { return nullptr; }

  void Start(ScorerUtterance * /*utterance*/, std::byte *state) override {
    lm::ngram::State begin;
    std::memset(&begin, 0, sizeof(begin));
    model_->BeginSentenceWrite(&begin);
    begin.ZeroRemaining();
    std::memcpy(state, &begin, sizeof(begin));
  }

 protected:
  ScorerAdvanceResult AdvanceOne(const ScorerExtension &extension) override {
    ScorerAdvanceResult result;
    result.log_score =
        Score(extension.parent.bytes, WordOf(extension.unit), extension.child);
    return result;
  }

  ScorerFinalScore FinishOne(const ScorerStateRef &state) override {
    ScorerFinalScore result;
    result.log_score = Score(state.bytes, end_sentence_, nullptr);
    return result;
  }

 private:
  // Natural-log score of "word" after "state". Writes the next state, with
  // unused bytes zeroed so equal contexts are equal bytes, unless out is null.
  float Score(const std::byte *state, lm::WordIndex word, std::byte *out) {
    lm::ngram::State next;
    std::memset(&next, 0, sizeof(next));
    float log10_prob = model_->BaseScore(state, word, &next);
    if (out != nullptr) {
      next.ZeroRemaining();
      std::memcpy(out, &next, sizeof(next));
    }
    return log10_prob * kLn10;
  }

  // The LM word for a unit; one the LM lacks is its <unk>.
  lm::WordIndex WordOf(UnitId unit) {
    if (info_.unit == ScorerUnit::kAsrToken && !ids_are_words_) {
      return static_cast<size_t>(unit) < word_of_token_.size()
                 ? word_of_token_[unit]
                 : model_->BaseVocabulary().NotFound();
    }
    int32_t known = word_of_.Lookup(unit);
    if (known >= 0) return static_cast<lm::WordIndex>(known);
    std::string text;
    if (ids_are_words_) {
      text = id_prefix_ + std::to_string(unit + id_offset_);
    } else {
      text = unit == ' ' ? space_ : Utf32ToUtf8(static_cast<char32_t>(unit));
    }
    lm::WordIndex word = model_->BaseVocabulary().Index(text);
    word_of_.Set(unit, static_cast<int32_t>(word));
    return word;
  }

  ScorerInfo info_;
  std::unique_ptr<lm::base::Model> model_;
  std::string space_;
  lm::WordIndex end_sentence_ = 0;
  bool ids_are_words_ = false;
  int32_t id_offset_ = 0;
  std::string id_prefix_;
  CharacterIndex word_of_;                    // Filled as units are first seen.
  std::vector<lm::WordIndex> word_of_token_;  // By token ID, for token units.
};

// The symbol of each token ID in a recognizer token table: lines of a
// symbol and its ID.
std::vector<std::string> ReadTokenSymbols(const std::string &filename) {
  std::ifstream is(filename);
  if (!is) {
    throw std::runtime_error("kenlm scorer: cannot read tokens '" + filename +
                             "'");
  }
  std::vector<std::string> symbols;
  std::string line;
  while (std::getline(is, line)) {
    size_t split = line.find_last_of(" \t");
    if (split == std::string::npos) continue;
    int32_t id = -1;
    try {
      id = std::stoi(line.substr(split + 1));
    } catch (const std::exception &) {
      continue;
    }
    if (id < 0) continue;
    if (symbols.size() <= static_cast<size_t>(id)) symbols.resize(id + 1);
    symbols[id] = line.substr(0, split);
  }
  if (symbols.empty()) {
    throw std::runtime_error("kenlm scorer: no tokens in '" + filename + "'");
  }
  return symbols;
}

class KenLmScorerFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions &common,
                                 const std::string &extra) const override {
    if (common.model.empty()) {
      throw std::runtime_error("kenlm scorer: needs a model path");
    }
    nlohmann::json options;
    try {
      options = nlohmann::json::parse(extra);
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("kenlm scorer: bad extra: ") +
                               e.what());
    }
    if (!options.is_object()) {
      throw std::runtime_error("kenlm scorer: extra must be a JSON object");
    }
    bool tokens = common.unit == "token";
    bool by_id = false;
    int32_t id_offset = 0;
    std::string id_prefix;
    std::vector<std::string> token_words;
    for (const auto &option : options.items()) {
      if (option.key() == "tokens" && option.value().is_string() && tokens) {
        token_words = ReadTokenSymbols(option.value().get<std::string>());
        continue;
      }
      if (option.key() == "id-offset" && option.value().is_number_integer() &&
          tokens) {
        by_id = true;
        id_offset = option.value().get<int32_t>();
        continue;
      }
      if (option.key() == "id-prefix" && option.value().is_string() && tokens) {
        by_id = true;
        id_prefix = option.value().get<std::string>();
        continue;
      }
      throw std::runtime_error("kenlm scorer: unknown or invalid option '" +
                               option.key() + "'");
    }
    if (tokens && by_id == !token_words.empty()) {
      throw std::runtime_error(
          "kenlm scorer: unit=token needs exactly one of extra \"tokens\", "
          "the recognizer's token table, and \"id-offset\" or \"id-prefix\"");
    }
    return std::make_unique<KenLmScorer>(common.model, common.space, tokens,
                                         token_words, id_offset, id_prefix);
  }
};

}  // namespace

std::unique_ptr<ScorerFactory> MakeKenLmScorerFactory() {
  return std::make_unique<KenLmScorerFactory>();
}

}  // namespace sherpa_onnx

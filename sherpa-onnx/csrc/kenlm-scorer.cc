// sherpa-onnx/csrc/kenlm-scorer.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/kenlm-scorer.h"

#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "lm/model.hh"
#include "lm/state.hh"
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

constexpr float kLn10 = 2.302585092994046f;

class KenLmScorer : public PerItemScorer {
 public:
  KenLmScorer(const std::string &model, const std::string &space)
      : space_(space) {
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
    info_.unit = ScorerUnit::kCharacter;
    info_.state_bytes = sizeof(lm::ngram::State);
    info_.recombines = true;
    end_sentence_ = model_->BaseVocabulary().EndSentence();
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

  // The LM word for a character; a character the LM lacks is its <unk>.
  lm::WordIndex WordOf(UnitId unit) {
    int32_t known = word_of_.Lookup(unit);
    if (known >= 0) return static_cast<lm::WordIndex>(known);
    std::string text =
        unit == ' ' ? space_ : Utf32ToUtf8(static_cast<char32_t>(unit));
    lm::WordIndex word = model_->BaseVocabulary().Index(text);
    word_of_.Set(unit, static_cast<int32_t>(word));
    return word;
  }

  ScorerInfo info_;
  std::unique_ptr<lm::base::Model> model_;
  std::string space_;
  lm::WordIndex end_sentence_ = 0;
  CharacterIndex word_of_;  // Filled as characters are first seen.
};

class KenLmScorerFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions &common,
                                 const std::string &extra) const override {
    if (common.unit != "char") {
      throw std::runtime_error(
          "kenlm scorer: only unit=char is supported. Given: " + common.unit);
    }
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
    for (const auto &option : options.items()) {
      throw std::runtime_error("kenlm scorer: unknown or invalid option '" +
                               option.key() + "'");
    }
    return std::make_unique<KenLmScorer>(common.model, common.space);
  }
};

}  // namespace

std::unique_ptr<ScorerFactory> MakeKenLmScorerFactory() {
  return std::make_unique<KenLmScorerFactory>();
}

}  // namespace sherpa_onnx

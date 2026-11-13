// sherpa-onnx/csrc/scorer.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/scorer.h"

#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/recurrent-lm-scorer.h"
#if SHERPA_ONNX_ENABLE_KENLM
#include "sherpa-onnx/csrc/kenlm-scorer.h"
#endif

namespace sherpa_onnx {

bool ParseScorerUnit(const std::string &name, ScorerUnit *unit) {
  if (name == "char") {
    *unit = ScorerUnit::kCharacter;
  } else if (name == "token") {
    *unit = ScorerUnit::kAsrToken;
  } else {
    return false;
  }
  return true;
}

namespace {

// Ordered so the "unknown type" message lists types predictably.
std::map<std::string, std::unique_ptr<ScorerFactory>> &ScorerFactories() {
  static auto *factories =
      new std::map<std::string, std::unique_ptr<ScorerFactory>>;
  return *factories;
}

}  // namespace

void RegisterScorerFactory(const std::string &type,
                           std::unique_ptr<ScorerFactory> factory) {
  if (type.empty() || factory == nullptr) {
    throw std::runtime_error("Scorer factory needs a type and a factory");
  }
  if (!ScorerFactories().emplace(type, std::move(factory)).second) {
    throw std::runtime_error("Scorer type registered twice: " + type);
  }
}

void RegisterBuiltinScorerFactories() {
  static std::once_flag once;
  std::call_once(once, [] {
    RegisterScorerFactory("onnx-rnn", MakeOnnxRecurrentLmScorerFactory());
#if SHERPA_ONNX_ENABLE_KENLM
    RegisterScorerFactory("kenlm", MakeKenLmScorerFactory());
#endif
  });
}

std::unique_ptr<Scorer> CreateScorer(const ScorerOptions &options) {
  RegisterBuiltinScorerFactories();
  const auto &factories = ScorerFactories();
  auto it = factories.find(options.type);
  if (it == factories.end()) {
    std::string known;
    for (const auto &entry : factories) {
      known += known.empty() ? "" : ", ";
      known += entry.first;
    }
    std::string hint;
#if !SHERPA_ONNX_ENABLE_KENLM
    if (options.type == "kenlm") {
      hint = " (kenlm needs a build with -DSHERPA_ONNX_ENABLE_KENLM=ON)";
    }
#endif
    throw std::runtime_error("Unknown scorer type '" + options.type +
                             "'. Registered types: " + known + hint);
  }
  return it->second->Create(options.common, options.extra);
}

std::vector<std::unique_ptr<Scorer>> CreateScorers(
    const std::vector<ScorerOptions> &options) {
  std::vector<std::unique_ptr<Scorer>> scorers;
  scorers.reserve(options.size());
  for (const auto &entry : options) {
    scorers.push_back(CreateScorer(entry));
  }
  return scorers;
}

}  // namespace sherpa_onnx

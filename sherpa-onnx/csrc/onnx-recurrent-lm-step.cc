// sherpa-onnx/csrc/onnx-recurrent-lm-step.cc
//
// Copyright (c)  2026  graehl

#include <array>
#include <exception>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/recurrent-lm-scorer.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

namespace {

class OnnxRecurrentLmStep : public RecurrentLmStep {
 public:
  explicit OnnxRecurrentLmStep(const CommonScorerOptions &common)
      : env_(ORT_LOGGING_LEVEL_ERROR),
        session_options_(GetSessionOptionsImpl(
            common.num_threads, common.provider.provider, &common.provider)) {
    auto model = ReadFile(common.model);
    if (model.empty()) {
      throw std::runtime_error("onnx-rnn scorer: cannot read model '" +
                               common.model + "'");
    }
    session_ = std::make_unique<Ort::Session>(env_, model.data(), model.size(),
                                              session_options_);
    GetInputNames(session_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(session_.get(), &output_names_, &output_names_ptr_);
    if (input_names_.size() < 2 ||
        output_names_.size() != input_names_.size()) {
      throw std::runtime_error(
          "onnx-rnn scorer: expected inputs (units, states...) and outputs "
          "(log probabilities, states...) in '" +
          common.model + "'");
    }
    for (size_t k = 1; k != input_names_.size(); ++k) {
      auto shape = session_->GetInputTypeInfo(k)
                       .GetTensorTypeAndShapeInfo()
                       .GetShape();  // [layers, batch, hidden]
      if (shape.size() != 3 || shape[0] <= 0 || shape[2] <= 0) {
        throw std::runtime_error(
            "onnx-rnn scorer: state input '" + input_names_[k] +
            "' is not [layers, batch, hidden] in '" + common.model + "'");
      }
      shapes_.push_back(
          {static_cast<int32_t>(shape[0]), static_cast<int32_t>(shape[2])});
    }
    auto shape = session_->GetOutputTypeInfo(0)
                     .GetTensorTypeAndShapeInfo()
                     .GetShape();  // [batch, vocab]
    if (shape.size() != 2 || shape[1] <= 0) {
      throw std::runtime_error(
          "onnx-rnn scorer: first output is not [batch, vocab] in '" +
          common.model + "'");
    }
    vocab_size_ = static_cast<int32_t>(shape[1]);
  }

  int32_t VocabSize() const override { return vocab_size_; }
  const std::vector<StateShape> &StateShapes() const override {
    return shapes_;
  }

  void Run(const std::vector<int64_t> &units,
           const std::vector<std::vector<float>> &states_in,
           const float **log_probs,
           std::vector<const float *> *states_out) override {
    auto memory =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    int64_t batch = static_cast<int64_t>(units.size());
    std::array<int64_t, 2> units_shape = {batch, 1};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory, const_cast<int64_t *>(units.data()), units.size(),
        units_shape.data(), units_shape.size()));
    for (size_t k = 0; k != shapes_.size(); ++k) {
      std::array<int64_t, 3> shape = {shapes_[k].layers, batch,
                                      shapes_[k].hidden};
      inputs.push_back(Ort::Value::CreateTensor<float>(
          memory, const_cast<float *>(states_in[k].data()), states_in[k].size(),
          shape.data(), shape.size()));
    }
    outputs_ =
        session_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                      output_names_ptr_.data(), output_names_ptr_.size());
    *log_probs = outputs_[0].GetTensorData<float>();
    states_out->resize(shapes_.size());
    for (size_t k = 0; k != shapes_.size(); ++k) {
      (*states_out)[k] = outputs_[k + 1].GetTensorData<float>();
    }
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
  std::vector<StateShape> shapes_;
  int32_t vocab_size_ = 0;
  std::vector<Ort::Value> outputs_;  // Kept until the next Run.
};

std::string Unhex(const std::string &hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

struct Extra {
  std::string vocab;
  bool hex = false;
  int32_t boundary_id = 0;
  int32_t unknown_id = 1;
  bool score_end = true;
  int32_t id_offset = 0;
};

Extra ParseExtra(const std::string &text, bool token_units) {
  Extra extra;
  if (token_units) {
    extra.boundary_id = 1;
    extra.unknown_id = 0;
  }
  nlohmann::json options;
  try {
    options = nlohmann::json::parse(text);
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("onnx-rnn scorer: bad extra: ") +
                             e.what());
  }
  if (!options.is_object()) {
    throw std::runtime_error("onnx-rnn scorer: extra must be a JSON object");
  }
  for (const auto &option : options.items()) {
    const std::string &key = option.key();
    const auto &value = option.value();
    if (key == "vocab" && value.is_string()) {
      extra.vocab = value.get<std::string>();
    } else if (key == "vocab-encoding" && value.is_string() &&
               (value == "utf8" || value == "hex")) {
      extra.hex = value == "hex";
    } else if (key == "boundary-id" && value.is_number_integer()) {
      extra.boundary_id = value.get<int32_t>();
    } else if (key == "unknown-id" && value.is_number_integer()) {
      extra.unknown_id = value.get<int32_t>();
    } else if (key == "id-offset" && value.is_number_integer() && token_units) {
      extra.id_offset = value.get<int32_t>();
    } else if (key == "score-end" && value.is_boolean()) {
      extra.score_end = value.get<bool>();
    } else {
      throw std::runtime_error("onnx-rnn scorer: unknown or invalid option '" +
                               key + "'");
    }
  }
  if (extra.vocab.empty() && !token_units) {
    throw std::runtime_error(
        "onnx-rnn scorer: extra needs \"vocab\", the unit list file");
  }
  return extra;
}

RecurrentLmVocabulary ReadVocabulary(const Extra &extra,
                                     const std::string &space,
                                     int32_t vocab_size) {
  std::ifstream is(extra.vocab);
  if (!is) {
    throw std::runtime_error("onnx-rnn scorer: cannot read vocab '" +
                             extra.vocab + "'");
  }
  RecurrentLmVocabulary vocabulary;
  vocabulary.boundary_id = extra.boundary_id;
  vocabulary.unknown_id = extra.unknown_id;
  std::string line;
  int32_t id = 0;
  for (; std::getline(is, line); ++id) {
    std::string unit = extra.hex ? Unhex(line) : line;
    if (unit == space) {
      vocabulary.id_of_char.Set(' ', id);
      continue;
    }
    std::u32string chars = Utf8ToUtf32(unit);
    // Entries that are not one character, such as "<unk>", are reached only
    // through boundary-id and unknown-id.
    if (chars.size() == 1 &&
        vocabulary.id_of_char.Lookup(static_cast<UnitId>(chars[0])) < 0) {
      vocabulary.id_of_char.Set(static_cast<UnitId>(chars[0]), id);
    }
  }
  if (id != vocab_size || extra.boundary_id < 0 ||
      extra.boundary_id >= vocab_size || extra.unknown_id < 0 ||
      extra.unknown_id >= vocab_size) {
    throw std::runtime_error(
        "onnx-rnn scorer: vocab '" + extra.vocab + "' has " +
        std::to_string(id) + " units but the model scores " +
        std::to_string(vocab_size) + ", or a special ID is out of range");
  }
  return vocabulary;
}

class OnnxRecurrentLmScorerFactory : public ScorerFactory {
 public:
  std::unique_ptr<Scorer> Create(const CommonScorerOptions &common,
                                 const std::string &extra_text) const override {
    bool token_units = common.unit == "token";
    Extra extra = ParseExtra(extra_text, token_units);
    auto step = std::make_unique<OnnxRecurrentLmStep>(common);
    RecurrentLmVocabulary vocabulary;
    if (token_units) {
      vocabulary.token_units = true;
      vocabulary.token_id_offset = extra.id_offset;
      vocabulary.boundary_id = extra.boundary_id;
      vocabulary.unknown_id = extra.unknown_id;
      if (extra.boundary_id < 0 || extra.boundary_id >= step->VocabSize() ||
          extra.unknown_id < 0 || extra.unknown_id >= step->VocabSize()) {
        throw std::runtime_error(
            "onnx-rnn scorer: a special ID is out of range");
      }
    } else {
      vocabulary = ReadVocabulary(extra, common.space, step->VocabSize());
    }
    vocabulary.score_end = extra.score_end;
    return MakeRecurrentLmScorer(std::move(step), std::move(vocabulary));
  }
};

}  // namespace

std::unique_ptr<ScorerFactory> MakeOnnxRecurrentLmScorerFactory() {
  return std::make_unique<OnnxRecurrentLmScorerFactory>();
}

}  // namespace sherpa_onnx

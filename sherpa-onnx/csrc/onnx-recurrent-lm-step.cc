// sherpa-onnx/csrc/onnx-recurrent-lm-step.cc
//
// Copyright (c)  2026  graehl

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <exception>
#include <fstream>
#include <limits>
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

    for (const StateShape &s : shapes_) {
      std::array<int64_t, 3> one = {s.layers, 1, s.hidden};
      zero_.push_back(Ort::Value::CreateTensor<float>(host_allocator_,
                                                      one.data(), one.size()));
      float *p = zero_.back().GetTensorMutableData<float>();
      std::fill(p, p + s.layers * s.hidden, 0.0f);
    }
    const std::string &provider = common.provider.provider;
    if ((provider == "cuda" || provider == "trt") && BuildGatherSessions()) {
      device_ = std::make_unique<Ort::MemoryInfo>("Cuda", OrtDeviceAllocator,
                                                  common.provider.device,
                                                  OrtMemTypeDefault);
    }
  }

  int32_t VocabSize() const override { return vocab_size_; }

  std::shared_ptr<const Result> Run(
      const std::vector<int64_t> &units,
      const std::vector<Parent> &parents) const override {
    auto host =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    int64_t batch = static_cast<int64_t>(units.size());
    std::array<int64_t, 2> units_shape = {batch, 1};
    Ort::Value units_in = Ort::Value::CreateTensor<int64_t>(
        host, const_cast<int64_t *>(units.data()), units.size(),
        units_shape.data(), units_shape.size());
    std::vector<Ort::Value> states_in =
        device_ ? DeviceGather(parents) : HostGather(parents);

    // The states stay where the model runs; the log probabilities come to
    // the host for the search.
    Ort::IoBinding binding(*session_);
    binding.BindInput(input_names_ptr_[0], units_in);
    for (size_t k = 0; k != shapes_.size(); ++k) {
      binding.BindInput(input_names_ptr_[k + 1], states_in[k]);
    }
    binding.BindOutput(output_names_ptr_[0], host);
    for (size_t k = 0; k != shapes_.size(); ++k) {
      binding.BindOutput(output_names_ptr_[k + 1], device_ ? *device_ : host);
    }
    session_->Run(Ort::RunOptions{}, binding);
    auto result = std::make_shared<StepResult>();
    result->values = binding.GetOutputValues();
    result->log_probs = result->values[0].GetTensorData<float>();
    result->vocab_size = vocab_size_;
    return result;
  }

 private:
  struct StateShape {
    int32_t layers = 0;
    int32_t hidden = 0;
  };

  // One Run's outputs: log probabilities [batch, vocab] on the host, then
  // each state [layers, batch, hidden] where the model runs.
  struct StepResult : public Result {
    std::vector<Ort::Value> values;
    const float *log_probs = nullptr;
    int32_t vocab_size = 0;

    const float *LogProbs(int32_t item) const override {
      return log_probs + static_cast<size_t>(item) * vocab_size;
    }
    const Ort::Value &State(size_t k) const { return values[k + 1]; }
  };

  // A state tensor of each source: the states of an earlier result, or
  // zero_ for the zero state.
  using Sources = std::vector<const std::vector<Ort::Value> *>;

  // The sources of the parents' states and, per parent, (source, item).
  void SourcesOf(const std::vector<Parent> &parents, Sources *sources,
                 std::vector<std::pair<int32_t, int32_t>> *picks) const {
    std::vector<const Result *> seen;
    for (const Parent &parent : parents) {
      int32_t j = 0;
      while (j != static_cast<int32_t>(seen.size()) &&
             seen[j] != parent.result) {
        ++j;
      }
      if (j == static_cast<int32_t>(seen.size())) {
        seen.push_back(parent.result);
        sources->push_back(StatesOf(parent.result));
      }
      picks->push_back({j, parent.result ? parent.item : 0});
    }
  }

  // An earlier result's state tensors, or the zero state's.
  const std::vector<Ort::Value> *StatesOf(const Result *result) const {
    if (result == nullptr) return &zero_;
    return &static_cast<const StepResult *>(result)->values;
  }

  // The state tensor of a source: values[k + 1] of a result, zero_[k].
  const Ort::Value &SourceState(const std::vector<Ort::Value> *source,
                                size_t k) const {
    return source == &zero_ ? zero_[k] : (*source)[k + 1];
  }

  // On the host: copies each parent's rows into one input per state.
  std::vector<Ort::Value> HostGather(const std::vector<Parent> &parents) const {
    Sources sources;
    std::vector<std::pair<int32_t, int32_t>> picks;
    SourcesOf(parents, &sources, &picks);
    const auto batch = static_cast<int64_t>(parents.size());
    std::vector<Ort::Value> states;
    for (size_t k = 0; k != shapes_.size(); ++k) {
      const int64_t layers = shapes_[k].layers, hidden = shapes_[k].hidden;
      std::array<int64_t, 3> shape = {layers, batch, hidden};
      states.push_back(Ort::Value::CreateTensor<float>(
          host_allocator_, shape.data(), shape.size()));
      float *to = states.back().GetTensorMutableData<float>();
      for (int64_t i = 0; i != batch; ++i) {
        const Ort::Value &from = SourceState(sources[picks[i].first], k);
        const int64_t width = from.GetTensorTypeAndShapeInfo().GetShape()[1];
        const float *p = from.GetTensorData<float>();
        for (int64_t layer = 0; layer != layers; ++layer) {
          std::copy_n(p + (layer * width + picks[i].second) * hidden, hidden,
                      to + (layer * batch + i) * hidden);
        }
      }
    }
    return states;
  }

  // On the device: gathers the parents' rows without leaving it.
  std::vector<Ort::Value> DeviceGather(
      const std::vector<Parent> &parents) const {
    Sources sources;
    std::vector<std::pair<int32_t, int32_t>> picks;
    SourcesOf(parents, &sources, &picks);
    std::deque<std::vector<Ort::Value>> held;  // intermediate levels
    return GatherRows(sources, picks, &held);
  }

  // Each state's rows picks[i] = (source, item) of `sources`, in one tensor
  // per state on the device. More sources than one gather graph takes are
  // gathered in groups first.
  std::vector<Ort::Value> GatherRows(
      const Sources &sources,
      const std::vector<std::pair<int32_t, int32_t>> &picks,
      std::deque<std::vector<Ort::Value>> *held) const {
    const auto num_sources = static_cast<int32_t>(sources.size());
    if (num_sources > kMaxSources) {
      Sources groups;
      std::vector<std::pair<int32_t, int32_t>> regrouped(picks.size());
      for (int32_t first = 0; first < num_sources; first += kMaxSources) {
        const int32_t end = std::min(num_sources, first + kMaxSources);
        Sources group(sources.begin() + first, sources.begin() + end);
        std::vector<std::pair<int32_t, int32_t>> group_picks;
        const auto g = static_cast<int32_t>(first / kMaxSources);
        for (size_t i = 0; i != picks.size(); ++i) {
          if (picks[i].first >= first && picks[i].first < end) {
            regrouped[i] = {g, static_cast<int32_t>(group_picks.size())};
            group_picks.push_back({picks[i].first - first, picks[i].second});
          }
        }
        // A group's output stands in for a result: its states at [k + 1].
        std::vector<Ort::Value> out;
        out.emplace_back(nullptr);
        for (auto &v : GatherRows(group, group_picks, held)) {
          out.push_back(std::move(v));
        }
        held->push_back(std::move(out));
        groups.push_back(&held->back());
      }
      return GatherRows(groups, regrouped, held);
    }

    auto host =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<int64_t> offsets(num_sources, 0);
    for (int32_t j = 1; j < num_sources; ++j) {
      offsets[j] = offsets[j - 1] + SourceState(sources[j - 1], 0)
                                        .GetTensorTypeAndShapeInfo()
                                        .GetShape()[1];
    }
    std::vector<int64_t> index(picks.size());
    for (size_t i = 0; i != picks.size(); ++i) {
      index[i] = offsets[picks[i].first] + picks[i].second;
    }
    std::array<int64_t, 1> index_shape = {static_cast<int64_t>(index.size())};
    Ort::Value index_in = Ort::Value::CreateTensor<int64_t>(
        host, index.data(), index.size(), index_shape.data(),
        index_shape.size());

    const GatherSession &gather = gather_[num_sources - 1];
    Ort::IoBinding binding(*gather.session);
    for (size_t k = 0; k != shapes_.size(); ++k) {
      for (int32_t j = 0; j != num_sources; ++j) {
        binding.BindInput(gather.input_names[k * num_sources + j].c_str(),
                          SourceState(sources[j], k));
      }
      binding.BindOutput(gather.output_names[k].c_str(), *device_);
    }
    binding.BindInput("index", index_in);
    gather.session->Run(Ort::RunOptions{}, binding);
    return binding.GetOutputValues();
  }

  // Builds, for 1 to kMaxSources sources, a graph that concatenates each
  // state's sources along the batch axis and gathers rows by index, to run
  // on the model's device. False when this onnxruntime cannot build graphs.
  bool BuildGatherSessions() {
#if ORT_API_VERSION >= 22
    if (Ort::GetApi().GetModelEditorApi() == nullptr) return false;
    const auto kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    auto info = [](const std::string &name, ONNXTensorElementDataType type,
                   const std::vector<int64_t> &dims) {
      Ort::TensorTypeAndShapeInfo shape(type, dims);
      Ort::TypeInfo type_info =
          Ort::TypeInfo::CreateTensorInfo(shape.GetConst());
      return Ort::ValueInfo(name, type_info.GetConst());
    };
    int64_t axis = 1;
    for (int32_t num_sources = 1; num_sources <= kMaxSources; ++num_sources) {
      GatherSession gather;
      Ort::Graph graph;
      std::vector<Ort::ValueInfo> inputs, outputs;
      for (size_t k = 0; k != shapes_.size(); ++k) {
        std::vector<int64_t> dims = {shapes_[k].layers, -1, shapes_[k].hidden};
        std::vector<std::string> names;
        for (int32_t j = 0; j != num_sources; ++j) {
          names.push_back("state" + std::to_string(k) + "_" +
                          std::to_string(j));
          gather.input_names.push_back(names.back());
          inputs.push_back(info(names.back(), kFloat, dims));
        }
        std::string all = names[0];
        if (num_sources > 1) {
          all = "all" + std::to_string(k);
          std::vector<Ort::OpAttr> attrs;
          attrs.emplace_back("axis", &axis, 1, ORT_OP_ATTR_INT);
          Ort::Node concat("Concat", "", "concat" + std::to_string(k), names,
                           {all}, attrs);
          graph.AddNode(concat);
        }
        gather.output_names.push_back("rows" + std::to_string(k));
        outputs.push_back(info(gather.output_names.back(), kFloat, dims));
        std::vector<Ort::OpAttr> attrs;
        attrs.emplace_back("axis", &axis, 1, ORT_OP_ATTR_INT);
        Ort::Node rows("Gather", "", "gather" + std::to_string(k),
                       {all, "index"}, {gather.output_names.back()}, attrs);
        graph.AddNode(rows);
      }
      inputs.push_back(
          info("index", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1}));
      graph.SetInputs(inputs);
      graph.SetOutputs(outputs);
      Ort::Model model({{"", 17}});
      model.AddGraph(graph);
      gather.session =
          std::make_unique<Ort::Session>(env_, model, session_options_);
      gather_.push_back(std::move(gather));
    }
    return true;
#else
    return false;
#endif
  }

  // A gather graph for a number of sources: inputs state<k>_<j> for each
  // state k and source j, then index; outputs rows<k>.
  struct GatherSession {
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
  };
  static constexpr int32_t kMaxSources = 8;

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;
  std::vector<StateShape> shapes_;
  int32_t vocab_size_ = 0;
  Ort::AllocatorWithDefaultOptions host_allocator_;
  // Each state [layers, 1, hidden] of zeros, on the host: the state before
  // the first unit.
  std::vector<Ort::Value> zero_;
  // Where the states stay, for a GPU provider; null keeps them on the host.
  std::unique_ptr<Ort::MemoryInfo> device_;
  std::vector<GatherSession> gather_;  // gather_[n - 1] takes n sources
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
  float score_floor = -std::numeric_limits<float>::infinity();
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
    } else if (key == "floor" && value.is_number()) {
      extra.score_floor = value.get<float>();
      if (!std::isfinite(extra.score_floor) || extra.score_floor > 0) {
        throw std::runtime_error(
            "onnx-rnn scorer: floor must be finite and nonpositive");
      }
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
    vocabulary.score_floor = extra.score_floor;
    return MakeRecurrentLmScorer(std::move(step), std::move(vocabulary));
  }
};

}  // namespace

std::unique_ptr<ScorerFactory> MakeOnnxRecurrentLmScorerFactory() {
  return std::make_unique<OnnxRecurrentLmScorerFactory>();
}

}  // namespace sherpa_onnx

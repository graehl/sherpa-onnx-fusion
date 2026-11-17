// sherpa-onnx/python/csrc/online-recongizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/online-recognizer.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/recognizer-options-json.h"

namespace sherpa_onnx {

namespace {

// The Python OnlineRecognizerConfig: the C++ config plus options_json, a
// JSON object of command-line options applied when a recognizer is created.
struct PyOnlineRecognizerConfig : public OnlineRecognizerConfig {
  std::string options_json;
};

OnlineRecognizerConfig ApplyOptionsJsonOrThrow(
    const PyOnlineRecognizerConfig &c) {
  OnlineRecognizerConfig config = c;
  std::string error;
  if (!ApplyOptionsJson(c.options_json, &config, &error)) {
    throw py::value_error("Error in options_json: " + error);
  }
  return config;
}

}  // namespace

static constexpr const char *kOnlineRecognizerResultDoc = R"doc(
Result for a streaming (online) recognizer. Call `result()` to obtain the
current partial/final decoding result for an ``OnlineStream``.
)doc";

static constexpr const char *kOnlineRecognizerConfigInitDoc = R"doc(
Configuration for the streaming (online) recognizer.

Args:
  feat_config:
    Config for the feature extractor.
  model_config:
    Config for the online model.
  lm_config:
    Config for the language model (optional).
  endpoint_config:
    Config for endpoint detection (optional).
  ctc_fst_decoder_config:
    Config for CTC FST decoding (optional).
  enable_endpoint:
    If True, enable endpoint detection.
  decoding_method:
    Decoding method, e.g., ``greedy_search`` or ``modified_beam_search``.
  max_active_paths:
    Number of active paths for modified beam search.
  hotwords_file:
    File containing hotwords, one per line.
  hotwords_score:
    Score for hotwords.
  blank_penalty:
    Penalty for blank tokens.
  temperature_scale:
    Temperature scale for non-blank tokens.
  rule_fsts:
    Rule FSTs for inverse text normalization.
  rule_fars:
    Rule FARs for inverse text normalization.
  reset_encoder:
    If True, reset the encoder state after endpoint detection.
  hr:
    Config for homophone replacer.
  fusion:
    Scorers and search options for ``decoding_method="prefix_beam_search"``.
  options_json:
    A JSON object of command-line options, applied after every typed field
    when the recognizer is created, e.g.
    ``{"scorer": [{"weight": 40}], "score-beam": 0}``. An error makes
    creation raise ``ValueError``.
)doc";

static constexpr const char *kOnlineRecognizerInitDoc = R"doc(
Constructor for the streaming (online) recognizer.

Args:
  config:
    The configuration for the recognizer.
)doc";

static constexpr const char *kCreateStreamDoc = R"doc(
Create a new ``OnlineStream`` for decoding.

Return:
  An ``OnlineStream`` object.
)doc";

static constexpr const char *kCreateStreamHotwordsDoc = R"doc(
Create a new ``OnlineStream`` for decoding with custom hotwords.

Args:
  hotwords:
    A string of hotwords separated by ``/``.
Return:
  An ``OnlineStream`` object.
)doc";

static constexpr const char *kIsReadyDoc = R"doc(
Check if the stream has enough frames for decoding.

Args:
  s:
    The ``OnlineStream`` to check.
Return:
  True if the stream has enough frames for decoding; False otherwise.
)doc";

static constexpr const char *kDecodeStreamDoc = R"doc(
Decode one frame for the given stream.

Args:
  s:
    The ``OnlineStream`` to decode.
)doc";

static constexpr const char *kDecodeStreamsDoc = R"doc(
Decode multiple streams at the same time.

Args:
  ss:
    A list of ``OnlineStream`` objects.
)doc";

static constexpr const char *kGetResultDoc = R"doc(
Get the current decoding result for the given stream.

Args:
  s:
    The ``OnlineStream``.
Return:
  An ``OnlineRecognizerResult`` object.
)doc";

static constexpr const char *kIsEndpointDoc = R"doc(
Check whether an endpoint has been detected for the given stream.

Args:
  s:
    The ``OnlineStream`` to check.
Return:
  True if an endpoint is detected; False otherwise.
)doc";

static constexpr const char *kResetDoc = R"doc(
Reset the given stream after an endpoint is detected. The internal state of
the stream is cleared.

Args:
  s:
    The ``OnlineStream`` to reset.
)doc";

static void PybindOnlineRecognizerResult(py::module *m) {
  using PyClass = OnlineRecognizerResult;
  py::class_<PyClass>(*m, "OnlineRecognizerResult", kOnlineRecognizerResultDoc)
      .def_property_readonly(
          "text",
          [](PyClass &self) -> py::str {
            return py::str(PyUnicode_DecodeUTF8(self.text.c_str(),
                                                self.text.size(), "ignore"));
          })
      .def_property_readonly(
          "tokens",
          [](PyClass &self) -> std::vector<std::string> { return self.tokens; })
      .def_property_readonly(
          "start_time", [](PyClass &self) -> float { return self.start_time; })
      .def_property_readonly(
          "timestamps",
          [](PyClass &self) -> std::vector<float> { return self.timestamps; })
      .def_property_readonly(
          "ys_probs",
          [](PyClass &self) -> std::vector<float> { return self.ys_probs; })
      .def_property_readonly(
          "lm_probs",
          [](PyClass &self) -> std::vector<float> { return self.lm_probs; })
      .def_property_readonly("context_scores",
                             [](PyClass &self) -> std::vector<float> {
                               return self.context_scores;
                             })
      .def_property_readonly(
          "segment", [](PyClass &self) -> int32_t { return self.segment; })
      .def_property_readonly(
          "words",
          [](PyClass &self) -> std::vector<int32_t> { return self.words; })
      .def_property_readonly(
          "is_final", [](PyClass &self) -> bool { return self.is_final; })
      .def("__str__", &PyClass::AsJsonString,
           py::call_guard<py::gil_scoped_release>())
      .def("as_json_string", &PyClass::AsJsonString,
           py::call_guard<py::gil_scoped_release>());
}

static void PybindOnlineRecognizerConfig(py::module *m) {
  using PyClass = PyOnlineRecognizerConfig;
  py::class_<PyClass>(*m, "OnlineRecognizerConfig")
      .def(py::init([](const FeatureExtractorConfig &feat_config,
                       const OnlineModelConfig &model_config,
                       const OnlineLMConfig &lm_config,
                       const EndpointConfig &endpoint_config,
                       const OnlineCtcFstDecoderConfig &ctc_fst_decoder_config,
                       bool enable_endpoint, const std::string &decoding_method,
                       int32_t max_active_paths,
                       const std::string &hotwords_file, float hotwords_score,
                       float blank_penalty, float temperature_scale,
                       const std::string &rule_fsts,
                       const std::string &rule_fars, bool reset_encoder,
                       const HomophoneReplacerConfig &hr,
                       const FusionConfig &fusion,
                       const std::string &options_json) {
             PyClass c;
             static_cast<OnlineRecognizerConfig &>(c) = OnlineRecognizerConfig(
                 feat_config, model_config, lm_config, endpoint_config,
                 ctc_fst_decoder_config, enable_endpoint, decoding_method,
                 max_active_paths, hotwords_file, hotwords_score, blank_penalty,
                 temperature_scale, rule_fsts, rule_fars, reset_encoder, hr);
             c.fusion = fusion;
             c.options_json = options_json;
             return c;
           }),
           py::arg("feat_config"), py::arg("model_config"),
           py::arg("lm_config") = OnlineLMConfig(),
           py::arg("endpoint_config") = EndpointConfig(),
           py::arg("ctc_fst_decoder_config") = OnlineCtcFstDecoderConfig(),
           py::arg("enable_endpoint"), py::arg("decoding_method"),
           py::arg("max_active_paths") = 4, py::arg("hotwords_file") = "",
           py::arg("hotwords_score") = 0, py::arg("blank_penalty") = 0.0,
           py::arg("temperature_scale") = 2.0, py::arg("rule_fsts") = "",
           py::arg("rule_fars") = "", py::arg("reset_encoder") = false,
           py::arg("hr") = HomophoneReplacerConfig{},
           py::arg("fusion") = FusionConfig{}, py::arg("options_json") = "",
           kOnlineRecognizerConfigInitDoc)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("endpoint_config", &PyClass::endpoint_config)
      .def_readwrite("ctc_fst_decoder_config", &PyClass::ctc_fst_decoder_config)
      .def_readwrite("enable_endpoint", &PyClass::enable_endpoint)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("hotwords_file", &PyClass::hotwords_file)
      .def_readwrite("hotwords_score", &PyClass::hotwords_score)
      .def_readwrite("blank_penalty", &PyClass::blank_penalty)
      .def_readwrite("temperature_scale", &PyClass::temperature_scale)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def_readwrite("reset_encoder", &PyClass::reset_encoder)
      .def_readwrite("hr", &PyClass::hr)
      .def_readwrite("fusion", &PyClass::fusion)
      .def_readwrite("options_json", &PyClass::options_json)
      .def("__str__", &PyClass::ToString);
}

void PybindOnlineRecognizer(py::module *m) {
  PybindOnlineRecognizerResult(m);
  PybindOnlineRecognizerConfig(m);

  using PyClass = OnlineRecognizer;
  py::class_<PyClass>(*m, "OnlineRecognizer")
      .def(py::init([](const PyOnlineRecognizerConfig &config) {
             return std::make_unique<PyClass>(ApplyOptionsJsonOrThrow(config));
           }),
           py::arg("config"), kOnlineRecognizerInitDoc,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          // A stream's prefix_beam_search state uses the recognizer's
          // scorers, so the stream keeps the recognizer alive.
          py::keep_alive<0, 1>(), kCreateStreamDoc,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &hotwords) {
            return self.CreateStream(hotwords);
          },
          py::arg("hotwords"), py::keep_alive<0, 1>(), kCreateStreamHotwordsDoc,
          py::call_guard<py::gil_scoped_release>())
      .def("is_ready", &PyClass::IsReady, kIsReadyDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, py::arg("s"),
           kDecodeStreamDoc, py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](PyClass &self, std::vector<OnlineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::arg("ss"), kDecodeStreamsDoc,
          py::call_guard<py::gil_scoped_release>())
      .def("get_result", &PyClass::GetResult, py::arg("s"), kGetResultDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("is_endpoint", &PyClass::IsEndpoint, py::arg("s"), kIsEndpointDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &PyClass::Reset, py::arg("s"), kResetDoc,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx

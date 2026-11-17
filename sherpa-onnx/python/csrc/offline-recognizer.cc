// sherpa-onnx/python/csrc/offline-recognizer.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-recognizer.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/recognizer-options-json.h"

namespace sherpa_onnx {

namespace {

// The Python OfflineRecognizerConfig: the C++ config plus options_json, a
// JSON object of command-line options applied when a recognizer is created.
struct PyOfflineRecognizerConfig : public OfflineRecognizerConfig {
  std::string options_json;
};

OfflineRecognizerConfig ApplyOptionsJsonOrThrow(
    const PyOfflineRecognizerConfig &c) {
  OfflineRecognizerConfig config = c;
  std::string error;
  if (!ApplyOptionsJson(c.options_json, &config, &error)) {
    throw py::value_error("Error in options_json: " + error);
  }
  return config;
}

}  // namespace

static constexpr const char *kOfflineRecognizerConfigInitDoc = R"doc(
Configuration for an offline (non-streaming) recognizer.

You can configure the model, language model, decoding method,
hotwords, and other recognition options here.

Args:
  feat_config:
    Config for the feature extractor (e.g., sample_rate, feature_dim).
  model_config:
    Config for the neural network model family (transducer, paraformer,
    whisper, etc.).
  lm_config:
    Config for the language model used during decoding.
  ctc_fst_decoder_config:
    Config for the CTC FST decoder.
  decoding_method:
    Decoding method to use. Supported values: ``greedy_search``,
    ``modified_beam_search``.
  max_active_paths:
    Number of active paths for beam search decoding. Only effective when
    ``decoding_method`` is ``modified_beam_search``.
  hotwords_file:
    Path to a file containing hotwords, one per line.
  hotwords_score:
    Score for hotwords. A larger value encourages the model to output
    the hotwords.
  blank_penalty:
    Penalty for blank tokens in CTC decoding.
  rule_fsts:
    Comma-separated list of rule FST filenames for inverse text
    normalization. They are applied from left to right.
  rule_fars:
    Comma-separated list of rule FAR (FST archive) filenames.
    They are applied from left to right.
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

static constexpr const char *kOfflineRecognizerInitDoc = R"doc(
Create an offline (non-streaming) recognizer.

Args:
  config:
    Configuration for the recognizer.
)doc";

static constexpr const char *kCreateStreamDoc = R"doc(
Create a new offline stream for decoding.

Return:
  An OfflineStream instance ready to accept audio.
)doc";

static constexpr const char *kCreateStreamHotwordsDoc = R"doc(
Create a new offline stream with custom hotwords for decoding.

Args:
  hotwords:
    A string containing hotwords separated by ``/``. Each hotword
    is composed of CJK characters or BPE tokens separated by spaces.
    For example, to boost ``I LOVE YOU`` and ``HELLO WORLD``::

      "▁I ▁LOVE ▁YOU/▁HE LL O ▁WORLD"

Return:
  An OfflineStream instance with the given hotwords.
)doc";

static constexpr const char *kDecodeStreamDoc = R"doc(
Run speech recognition on a single stream.

Args:
  s:
    The stream to decode. You must call ``accept_waveform`` on the
    stream before calling this method.
)doc";

static constexpr const char *kDecodeStreamsDoc = R"doc(
Run speech recognition on multiple streams in parallel.

Args:
  ss:
    A list of OfflineStream instances to decode.
)doc";

static constexpr const char *kEncodeStreamDoc = R"doc(
Run the model for a stream ahead of decoding it; ``decode_stream`` then
only searches. Optional: decoding runs the model for a stream that was not
encoded. Does nothing for a model family that does not separate the two
steps (every family except CTC, the zipformer transducer and NeMo
transducers).

Args:
  s:
    The stream to encode, after ``accept_waveform``.
)doc";

static constexpr const char *kEncodeStreamsDoc = R"doc(
Run the model for a list of streams ahead of decoding them. The streams
encoded together and those decoded together need not be the same.

Args:
  ss:
    A list of OfflineStream instances to encode.
)doc";

static constexpr const char *kSetConfigDoc = R"doc(
Update the recognizer configuration at runtime.

Args:
  config:
    The new configuration to apply. Only some fields (e.g., language,
    task for Whisper models) take effect after creation.
)doc";

static void PybindOfflineRecognizerConfig(py::module *m) {
  using PyClass = PyOfflineRecognizerConfig;
  py::class_<PyClass>(*m, "OfflineRecognizerConfig")
      .def(py::init(
               [](const FeatureExtractorConfig &feat_config,
                  const OfflineModelConfig &model_config,
                  const OfflineLMConfig &lm_config,
                  const OfflineCtcFstDecoderConfig &ctc_fst_decoder_config,
                  const std::string &decoding_method, int32_t max_active_paths,
                  const std::string &hotwords_file, float hotwords_score,
                  float blank_penalty, const std::string &rule_fsts,
                  const std::string &rule_fars,
                  const HomophoneReplacerConfig &hr, const FusionConfig &fusion,
                  const std::string &options_json) {
                 PyClass c;
                 static_cast<OfflineRecognizerConfig &>(c) =
                     OfflineRecognizerConfig(
                         feat_config, model_config, lm_config,
                         ctc_fst_decoder_config, decoding_method,
                         max_active_paths, hotwords_file, hotwords_score,
                         blank_penalty, rule_fsts, rule_fars, hr);
                 c.fusion = fusion;
                 c.options_json = options_json;
                 return c;
               }),
           py::arg("feat_config") = FeatureExtractorConfig(),
           py::arg("model_config") = OfflineModelConfig(),
           py::arg("lm_config") = OfflineLMConfig(),
           py::arg("ctc_fst_decoder_config") = OfflineCtcFstDecoderConfig(),
           py::arg("decoding_method") = "greedy_search",
           py::arg("max_active_paths") = 4, py::arg("hotwords_file") = "",
           py::arg("hotwords_score") = 1.5, py::arg("blank_penalty") = 0.0,
           py::arg("rule_fsts") = "", py::arg("rule_fars") = "",
           py::arg("hr") = HomophoneReplacerConfig{},
           py::arg("fusion") = FusionConfig{}, py::arg("options_json") = "",
           kOfflineRecognizerConfigInitDoc)
      .def_readwrite("feat_config", &PyClass::feat_config)
      .def_readwrite("model_config", &PyClass::model_config)
      .def_readwrite("lm_config", &PyClass::lm_config)
      .def_readwrite("ctc_fst_decoder_config", &PyClass::ctc_fst_decoder_config)
      .def_readwrite("decoding_method", &PyClass::decoding_method)
      .def_readwrite("max_active_paths", &PyClass::max_active_paths)
      .def_readwrite("hotwords_file", &PyClass::hotwords_file)
      .def_readwrite("hotwords_score", &PyClass::hotwords_score)
      .def_readwrite("blank_penalty", &PyClass::blank_penalty)
      .def_readwrite("rule_fsts", &PyClass::rule_fsts)
      .def_readwrite("rule_fars", &PyClass::rule_fars)
      .def_readwrite("hr", &PyClass::hr)
      .def_readwrite("fusion", &PyClass::fusion)
      .def_readwrite("options_json", &PyClass::options_json)
      .def("__str__", &PyClass::ToString);
}

void PybindOfflineRecognizer(py::module *m) {
  PybindOfflineRecognizerConfig(m);

  using PyClass = OfflineRecognizer;
  py::class_<PyClass>(*m, "OfflineRecognizer")
      .def(py::init([](const PyOfflineRecognizerConfig &config) {
             return std::make_unique<PyClass>(ApplyOptionsJsonOrThrow(config));
           }),
           py::arg("config"), kOfflineRecognizerInitDoc,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](const PyClass &self) { return self.CreateStream(); },
          kCreateStreamDoc, py::call_guard<py::gil_scoped_release>())
      .def(
          "create_stream",
          [](PyClass &self, const std::string &hotwords) {
            return self.CreateStream(hotwords);
          },
          py::arg("hotwords"), kCreateStreamHotwordsDoc,
          py::call_guard<py::gil_scoped_release>())
      .def("decode_stream", &PyClass::DecodeStream, py::arg("s"),
           kDecodeStreamDoc, py::call_guard<py::gil_scoped_release>())
      .def(
          "set_config",
          [](PyClass &self, const PyOfflineRecognizerConfig &config) {
            self.SetConfig(ApplyOptionsJsonOrThrow(config));
          },
          py::arg("config"), kSetConfigDoc,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "decode_streams",
          [](const PyClass &self, std::vector<OfflineStream *> ss) {
            self.DecodeStreams(ss.data(), ss.size());
          },
          py::arg("ss"), kDecodeStreamsDoc,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "encode_stream",
          [](const PyClass &self, OfflineStream *s) {
            self.EncodeStreams(&s, 1);
          },
          py::arg("s"), kEncodeStreamDoc,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "encode_streams",
          [](const PyClass &self, std::vector<OfflineStream *> ss) {
            self.EncodeStreams(ss.data(), ss.size());
          },
          py::arg("ss"), kEncodeStreamsDoc,
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx

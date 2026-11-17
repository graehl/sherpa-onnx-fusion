// sherpa-onnx/python/csrc/offline-model-config.cc
//
// Copyright (c)  2023 by manyeyes

#include "sherpa-onnx/python/csrc/offline-model-config.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/python/csrc/offline-canary-model-config.h"
#include "sherpa-onnx/python/csrc/offline-cohere-transcribe-model-config.h"
#include "sherpa-onnx/python/csrc/offline-dolphin-model-config.h"
#include "sherpa-onnx/python/csrc/offline-encoder-config.h"
#include "sherpa-onnx/python/csrc/offline-fire-red-asr-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-fire-red-asr-model-config.h"
#include "sherpa-onnx/python/csrc/offline-funasr-nano-model-config.h"
#include "sherpa-onnx/python/csrc/offline-medasr-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-moonshine-model-config.h"
#include "sherpa-onnx/python/csrc/offline-nemo-enc-dec-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-omnilingual-asr-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-paraformer-model-config.h"
#include "sherpa-onnx/python/csrc/offline-qwen3-asr-model-config.h"
#include "sherpa-onnx/python/csrc/offline-sense-voice-model-config.h"
#include "sherpa-onnx/python/csrc/offline-tdnn-model-config.h"
#include "sherpa-onnx/python/csrc/offline-transducer-model-config.h"
#include "sherpa-onnx/python/csrc/offline-wenet-ctc-model-config.h"
#include "sherpa-onnx/python/csrc/offline-whisper-model-config.h"
#include "sherpa-onnx/python/csrc/offline-zipformer-ctc-model-config.h"

namespace sherpa_onnx {

static constexpr const char *kOfflineModelConfigInitDoc = R"doc(
Configuration for the offline ASR model.

You only need to fill in the sub-config that matches the model family you
are using. For instance, if you are using a Whisper model, configure
``whisper`` and leave the other model configs at their defaults.

Args:
  transducer:
    Config for a transducer (e.g., Zipformer transducer) model.
  paraformer:
    Config for a Paraformer model.
  nemo_ctc:
    Config for a NeMo EncDec CTC model.
  whisper:
    Config for a Whisper model.
  fire_red_asr:
    Config for a FireRedAsr model.
  tdnn:
    Config for a TDNN model.
  zipformer_ctc:
    Config for a Zipformer CTC model.
  wenet_ctc:
    Config for a WeNet CTC model.
  sense_voice:
    Config for a SenseVoice model.
  moonshine:
    Config for a Moonshine model.
  dolphin:
    Config for a Dolphin model.
  canary:
    Config for a Canary model.
  cohere_transcribe:
    Config for a Cohere Transcribe model.
  omnilingual:
    Config for an Omnilingual ASR CTC model.
  funasr_nano:
    Config for a FunASR Nano model.
  medasr:
    Config for a MedASR CTC model.
  fire_red_asr_ctc:
    Config for a FireRedAsr CTC model.
  qwen3_asr:
    Config for a Qwen3 ASR model.
  telespeech_ctc:
    Path to the TeleSpeech CTC ONNX model file.
  tokens:
    Path to the tokens file for the model.
  num_threads:
    Number of threads for ONNX Runtime inference.
  debug:
    If True, print debug information during model loading.
  provider:
    The ONNX Runtime provider to use. Supported values:
    ``cpu``, ``cuda``, ``coreml``, ``xnnpack``, etc.
  model_type:
    Type of the model. If not set, it will be inferred from the
    model filename.
  modeling_unit:
    The modeling unit used by the model (e.g., ``cjkchar``, ``bpe``).
  bpe_vocab:
    Path to the BPE vocabulary file.
  w2v_bert_ctc:
    Path to a w2v-BERT 2.0 CTC model exported from transformers, with the
    single input ``input_features``.
  litert_dispatch_dir:
    Directory of LiteRT's accelerator library (for a Pixel,
    ``libLiteRtDispatch_GoogleTensor.so``), for a ``.tflite`` model run with
    ``provider="npu"``.
  litert_window:
    ``WINDOW_S:OVERLAP_S:SLACK_S``: a ``.tflite`` model runs an utterance
    longer than ``WINDOW_S`` in blended overlapping windows of its shortest
    length of at least ``WINDOW_S``; empty: of its longest.
  encoder:
    An :class:`OfflineEncoderConfig`: how the encoder runs, e.g. a long
    utterance in windows.
)doc";

void PybindOfflineModelConfig(py::module *m) {
  PybindOfflineTransducerModelConfig(m);
  PybindOfflineParaformerModelConfig(m);
  PybindOfflineNemoEncDecCtcModelConfig(m);
  PybindOfflineWhisperModelConfig(m);
  PybindOfflineFireRedAsrModelConfig(m);
  PybindOfflineTdnnModelConfig(m);
  PybindOfflineZipformerCtcModelConfig(m);
  PybindOfflineWenetCtcModelConfig(m);
  PybindOfflineSenseVoiceModelConfig(m);
  PybindOfflineMoonshineModelConfig(m);
  PybindOfflineDolphinModelConfig(m);
  PybindOfflineCanaryModelConfig(m);
  PybindOfflineCohereTranscribeModelConfig(m);
  PybindOfflineOmnilingualAsrCtcModelConfig(m);
  PybindOfflineFunASRNanoModelConfig(m);
  PybindOfflineMedAsrCtcModelConfig(m);
  PybindOfflineFireRedAsrCtcModelConfig(m);
  PybindOfflineQwen3ASRModelConfig(m);
  PybindOfflineEncoderConfig(m);

  using PyClass = OfflineModelConfig;
  py::class_<PyClass>(*m, "OfflineModelConfig")
      .def(
          py::init(
              [](const OfflineTransducerModelConfig &transducer,
                 const OfflineParaformerModelConfig &paraformer,
                 const OfflineNemoEncDecCtcModelConfig &nemo_ctc,
                 const OfflineWhisperModelConfig &whisper,
                 const OfflineFireRedAsrModelConfig &fire_red_asr,
                 const OfflineTdnnModelConfig &tdnn,
                 const OfflineZipformerCtcModelConfig &zipformer_ctc,
                 const OfflineWenetCtcModelConfig &wenet_ctc,
                 const OfflineSenseVoiceModelConfig &sense_voice,
                 const OfflineMoonshineModelConfig &moonshine,
                 const OfflineDolphinModelConfig &dolphin,
                 const OfflineCanaryModelConfig &canary,
                 const OfflineCohereTranscribeModelConfig &cohere_transcribe,
                 const OfflineOmnilingualAsrCtcModelConfig &omnilingual,
                 const OfflineFunASRNanoModelConfig &funasr_nano,
                 const OfflineMedAsrCtcModelConfig &medasr,
                 const OfflineFireRedAsrCtcModelConfig &fire_red_asr_ctc,
                 const OfflineQwen3ASRModelConfig &qwen3_asr,
                 const std::string &telespeech_ctc, const std::string &tokens,
                 int32_t num_threads, bool debug, const std::string &provider,
                 const std::string &model_type,
                 const std::string &modeling_unit, const std::string &bpe_vocab,
                 const std::string &w2v_bert_ctc,
                 const std::string &litert_dispatch_dir,
                 const std::string &litert_window,
                 const OfflineEncoderConfig &encoder) {
                PyClass c(transducer, paraformer, nemo_ctc, whisper,
                          fire_red_asr, tdnn, zipformer_ctc, wenet_ctc,
                          sense_voice, moonshine, dolphin, canary,
                          cohere_transcribe, omnilingual, funasr_nano, medasr,
                          fire_red_asr_ctc, qwen3_asr, telespeech_ctc, tokens,
                          num_threads, debug, provider, model_type,
                          modeling_unit, bpe_vocab);
                c.w2v_bert_ctc = w2v_bert_ctc;
                c.litert_dispatch_dir = litert_dispatch_dir;
                c.litert_window = litert_window;
                c.encoder = encoder;
                return c;
              }),
          py::arg("transducer") = OfflineTransducerModelConfig(),
          py::arg("paraformer") = OfflineParaformerModelConfig(),
          py::arg("nemo_ctc") = OfflineNemoEncDecCtcModelConfig(),
          py::arg("whisper") = OfflineWhisperModelConfig(),
          py::arg("fire_red_asr") = OfflineFireRedAsrModelConfig(),
          py::arg("tdnn") = OfflineTdnnModelConfig(),
          py::arg("zipformer_ctc") = OfflineZipformerCtcModelConfig(),
          py::arg("wenet_ctc") = OfflineWenetCtcModelConfig(),
          py::arg("sense_voice") = OfflineSenseVoiceModelConfig(),
          py::arg("moonshine") = OfflineMoonshineModelConfig(),
          py::arg("dolphin") = OfflineDolphinModelConfig(),
          py::arg("canary") = OfflineCanaryModelConfig(),
          py::arg("cohere_transcribe") = OfflineCohereTranscribeModelConfig(),
          py::arg("omnilingual") = OfflineOmnilingualAsrCtcModelConfig(),
          py::arg("funasr_nano") = OfflineFunASRNanoModelConfig(),
          py::arg("medasr") = OfflineMedAsrCtcModelConfig(),
          py::arg("fire_red_asr_ctc") = OfflineFireRedAsrCtcModelConfig(),
          py::arg("qwen3_asr") = OfflineQwen3ASRModelConfig(),
          py::arg("telespeech_ctc") = "", py::arg("tokens") = "",
          py::arg("num_threads") = 1, py::arg("debug") = false,
          py::arg("provider") = "cpu", py::arg("model_type") = "",
          py::arg("modeling_unit") = "cjkchar", py::arg("bpe_vocab") = "",
          py::arg("w2v_bert_ctc") = "", py::arg("litert_dispatch_dir") = "",
          py::arg("litert_window") = "",
          py::arg("encoder") = OfflineEncoderConfig{},
          kOfflineModelConfigInitDoc)
      .def_readwrite("transducer", &PyClass::transducer)
      .def_readwrite("paraformer", &PyClass::paraformer)
      .def_readwrite("nemo_ctc", &PyClass::nemo_ctc)
      .def_readwrite("whisper", &PyClass::whisper)
      .def_readwrite("fire_red_asr", &PyClass::fire_red_asr)
      .def_readwrite("tdnn", &PyClass::tdnn)
      .def_readwrite("zipformer_ctc", &PyClass::zipformer_ctc)
      .def_readwrite("wenet_ctc", &PyClass::wenet_ctc)
      .def_readwrite("sense_voice", &PyClass::sense_voice)
      .def_readwrite("moonshine", &PyClass::moonshine)
      .def_readwrite("dolphin", &PyClass::dolphin)
      .def_readwrite("canary", &PyClass::canary)
      .def_readwrite("cohere_transcribe", &PyClass::cohere_transcribe)
      .def_readwrite("omnilingual", &PyClass::omnilingual)
      .def_readwrite("funasr_nano", &PyClass::funasr_nano)
      .def_readwrite("medasr", &PyClass::medasr)
      .def_readwrite("fire_red_asr_ctc", &PyClass::fire_red_asr_ctc)
      .def_readwrite("qwen3_asr", &PyClass::qwen3_asr)
      .def_readwrite("telespeech_ctc", &PyClass::telespeech_ctc)
      .def_readwrite("w2v_bert_ctc", &PyClass::w2v_bert_ctc)
      .def_readwrite("litert_dispatch_dir", &PyClass::litert_dispatch_dir)
      .def_readwrite("litert_window", &PyClass::litert_window)
      .def_readwrite("encoder", &PyClass::encoder)
      .def_readwrite("tokens", &PyClass::tokens)
      .def_readwrite("num_threads", &PyClass::num_threads)
      .def_readwrite("debug", &PyClass::debug)
      .def_readwrite("provider", &PyClass::provider)
      .def_readwrite("model_type", &PyClass::model_type)
      .def_readwrite("modeling_unit", &PyClass::modeling_unit)
      .def_readwrite("bpe_vocab", &PyClass::bpe_vocab)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

}  // namespace sherpa_onnx

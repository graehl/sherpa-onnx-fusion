// c-api-examples/prefix-beam-search-lm-c-api.c
//
// Copyright (c)  2026  graehl

//
// This file demonstrates prefix beam search with a character language model
// fused into the search, using sherpa-onnx's C API. Needs a build with
// -DSHERPA_ONNX_ENABLE_KENLM=ON.
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// wget https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/en-librispeech-char-8gram.bin
//
// clang-format on

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

int32_t main() {
  // clang-format off
  const char *wav_filename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav";
  const char *model_filename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx";
  const char *tokens_filename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt";
  const char *lm_filename = "./en-librispeech-char-8gram.bin";
  // clang-format on

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(wav_filename);
  if (wave == NULL) {
    fprintf(stderr, "Failed to read %s\n", wav_filename);
    return -1;
  }

  SherpaOnnxOfflineRecognizerConfig recognizer_config;
  memset(&recognizer_config, 0, sizeof(recognizer_config));
  recognizer_config.model_config.num_threads = 1;
  recognizer_config.model_config.provider = "cpu";
  recognizer_config.model_config.tokens = tokens_filename;
  recognizer_config.model_config.zipformer_ctc.model = model_filename;
  recognizer_config.decoding_method = "prefix_beam_search";

  // One scorer; several may be given, each with its own weight.
  SherpaOnnxScorerConfig lm;
  memset(&lm, 0, sizeof(lm));
  lm.type = "kenlm";
  lm.model = lm_filename;
  lm.weight = 0.6f;
  recognizer_config.fusion.scorers = &lm;
  recognizer_config.fusion.num_scorers = 1;

  // Any command-line option can also be given as JSON, applied after the
  // fields above; here it changes the first scorer's weight.
  recognizer_config.options_json = "{\"scorer\": [{\"weight\": 0.5}]}";

  const SherpaOnnxOfflineRecognizer *recognizer =
      SherpaOnnxCreateOfflineRecognizer(&recognizer_config);

  if (recognizer == NULL) {
    fprintf(stderr, "Please check your config!\n");
    SherpaOnnxFreeWave(wave);
    return -1;
  }

  const SherpaOnnxOfflineStream *stream =
      SherpaOnnxCreateOfflineStream(recognizer);

  SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                  wave->num_samples);
  SherpaOnnxDecodeOfflineStream(recognizer, stream);
  const SherpaOnnxOfflineRecognizerResult *result =
      SherpaOnnxGetOfflineStreamResult(stream);

  fprintf(stderr, "Decoded text: %s\n", result->text);

  SherpaOnnxDestroyOfflineRecognizerResult(result);
  SherpaOnnxDestroyOfflineStream(stream);
  SherpaOnnxDestroyOfflineRecognizer(recognizer);
  SherpaOnnxFreeWave(wave);

  return 0;
}

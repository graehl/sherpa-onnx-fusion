// Copyright (c)  2026  Xiaomi Corporation
//
// Prefix beam search over a zipformer CTC model with a KenLM language model
// fused into the search, e.g.
//
//   dart run ./bin/zipformer-ctc-fusion.dart \
//     --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
//     --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
//     --lm ./en-librispeech-char-6gram.bin \
//     --input-wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav
//
// --options-json takes any further command-line option of
// sherpa-onnx-offline, e.g. '{"scorer": [{"weight": 0.8}], "score-beam": 16}'.
import 'dart:io';

import 'package:args/args.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main(List<String> arguments) async {
  await sherpa_onnx.initBindingsAsync();

  final parser = ArgParser()
    ..addOption('model', help: 'Path to the Zipformer CTC model')
    ..addOption('tokens', help: 'Path to tokens.txt')
    ..addOption('lm', help: 'Path to a character KenLM model')
    ..addOption('lm-weight', help: 'Weight of the LM', defaultsTo: '0.6')
    ..addOption(
      'options-json',
      help: 'JSON object of further options',
      defaultsTo: '',
    )
    ..addOption('input-wav', help: 'Path to input.wav to transcribe');

  final res = parser.parse(arguments);
  if (res['model'] == null ||
      res['tokens'] == null ||
      res['lm'] == null ||
      res['input-wav'] == null) {
    print(parser.usage);
    exit(1);
  }

  final model = res['model'] as String;
  final tokens = res['tokens'] as String;
  final lm = res['lm'] as String;
  final lmWeight = double.parse(res['lm-weight'] as String);
  final optionsJson = res['options-json'] as String;
  final inputWav = res['input-wav'] as String;

  final modelConfig = sherpa_onnx.OfflineModelConfig(
    zipformerCtc: sherpa_onnx.OfflineZipformerCtcModelConfig(model: model),
    tokens: tokens,
    debug: false,
    numThreads: 1,
  );

  final fusion = sherpa_onnx.FusionConfig(
    scorers: [
      sherpa_onnx.ScorerConfig(type: 'kenlm', model: lm, weight: lmWeight),
    ],
  );

  final config = sherpa_onnx.OfflineRecognizerConfig(
    model: modelConfig,
    decodingMethod: 'prefix_beam_search',
    fusion: fusion,
    optionsJson: optionsJson,
  );
  final recognizer = sherpa_onnx.OfflineRecognizer(config);

  final waveData = sherpa_onnx.readWave(inputWav);
  final stream = recognizer.createStream();

  stream.acceptWaveform(
    samples: waveData.samples,
    sampleRate: waveData.sampleRate,
  );
  recognizer.decode(stream);

  final result = recognizer.getResult(stream);
  print(result.text);

  stream.free();
  recognizer.free();
}

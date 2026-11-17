// Copyright (c)  2026  graehl
//
// Prefix beam search with a KenLM character language model fused into the
// search, on an English zipformer CTC model.
//
// Please download test files:
// https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/en-librispeech-char-6gram.bin
const sherpa_onnx = require('sherpa-onnx-node');

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'zipformerCtc': {
      'model': './sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx',
    },
    'tokens': './sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 0,
  },
  'decodingMethod': 'prefix_beam_search',
  'fusion': {
    'scorers': [{
      'type': 'kenlm',
      'model': './en-librispeech-char-6gram.bin',
      'weight': 0.35,
    }],
  },
  // Options the typed fields above cannot express, with the names that
  // sherpa-onnx-offline --help prints; applied after every other field.
  'optionsJson': '{"length-bonus": 0.5}',
};

const waveFilename = './sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started');
let start = Date.now();
const stream = recognizer.createStream();
const wave = sherpa_onnx.readWave(waveFilename);
stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});

recognizer.decode(stream);
const result = recognizer.getResult(stream);
let stop = Date.now();
console.log('Done');

const elapsed_seconds = (stop - start) / 1000;
const duration = wave.samples.length / wave.sampleRate;
const real_time_factor = elapsed_seconds / duration;
console.log('Wave duration', duration.toFixed(3), 'seconds');
console.log('Elapsed', elapsed_seconds.toFixed(3), 'seconds');
console.log(
    `RTF = ${elapsed_seconds.toFixed(3)}/${duration.toFixed(3)} =`,
    real_time_factor.toFixed(3));
console.log(waveFilename);
console.log('result\n', result);

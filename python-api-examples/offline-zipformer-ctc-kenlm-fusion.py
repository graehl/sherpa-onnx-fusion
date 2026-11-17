#!/usr/bin/env python3

# This file shows how to decode with prefix_beam_search and a KenLM scorer
# fused into the search, using a non-streaming zipformer CTC model and
# sherpa-onnx's Python API. options_json sets any further command-line
# option of the recognizer, here a larger LM weight and no score beam.
#
# clang-format off
#
# wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
# tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
#
# python3 ./python-api-examples/offline-zipformer-ctc-kenlm-fusion.py \
#   --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
#   --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
#   --lm ./en-librispeech-char-6gram.bin \
#   --options-json '{"scorer": [{"weight": 0.8}], "score-beam": 0}' \
#   ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav
#
# clang-format on

import argparse
import wave

import sherpa_onnx


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tokens", required=True, help="Path to tokens.txt")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    parser.add_argument("--lm", required=True, help="Path to a KenLM model")
    parser.add_argument(
        "--lm-unit", default="char", help="What the LM scores: char or token"
    )
    parser.add_argument("--lm-weight", type=float, default=0.6)
    parser.add_argument(
        "--options-json",
        default="",
        help="A JSON object of further command-line options",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("sound_file", help="A 16-bit mono WAVE file")
    return parser.parse_args()


def read_wave(filename: str):
    with wave.open(filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        data = f.readframes(f.getnframes())
        rate = f.getframerate()
    return [s / 32768 for s in memoryview(data).cast("h")], rate


def main():
    args = get_args()
    fusion = sherpa_onnx.FusionConfig(
        scorers=[
            sherpa_onnx.ScorerConfig(
                type="kenlm", model=args.lm, unit=args.lm_unit, weight=args.lm_weight
            )
        ]
    )
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=args.model,
        tokens=args.tokens,
        num_threads=args.num_threads,
        decoding_method="prefix_beam_search",
        fusion=fusion,
        options_json=args.options_json,
    )

    samples, sample_rate = read_wave(args.sound_file)
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    print(stream.result.text)


if __name__ == "__main__":
    main()

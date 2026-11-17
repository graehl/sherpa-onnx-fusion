#!/usr/bin/env bash
set -ex

if [ ! -f "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx" ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  ls -lh sherpa-onnx-zipformer-ctc-en-2023-10-02
fi

if [ ! -f ./en-librispeech-char-6gram.bin ]; then
  curl -SL -O https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/en-librispeech-char-6gram.bin
fi

# prefix_beam_search with a KenLM model fused into the search; options_json
# sets what the typed fields do not.
cargo run --example zipformer_ctc_fusion -- \
    --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
    --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
    --lm ./en-librispeech-char-6gram.bin \
    --options-json '{"length-bonus": 0.5}' \
    --wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav

#!/usr/bin/env bash

set -ex

export CGO_ENABLED=1

m=sherpa-onnx-zipformer-ctc-en-2023-10-02
if [ ! -f ./$m/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$m.tar.bz2
  tar xvf $m.tar.bz2
  rm $m.tar.bz2
fi

lm=en-librispeech-char-6gram.bin
if [ ! -f ./$lm ]; then
  curl -SL -O https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/$lm
fi

go mod tidy
go build

./non-streaming-fusion-decode-files \
  --zipformer-ctc ./$m/model.int8.onnx \
  --tokens ./$m/tokens.txt \
  --lm ./$lm \
  --lm-weight 0.6 \
  --options-json '{"length-bonus": 0.5}' \
  ./$m/test_wavs/1.wav

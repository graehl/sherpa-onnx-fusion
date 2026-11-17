#!/usr/bin/env bash
#
# Needs ../build to be configured with -DSHERPA_ONNX_ENABLE_KENLM=ON.

set -ex

source ./setup.sh

if [ ! -f ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2

  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
fi

if [ ! -f ./en-librispeech-char-6gram.bin ]; then
  curl -SL -O https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/en-librispeech-char-6gram.bin
fi

java \
  -Dsherpa_onnx.native.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  NonStreamingDecodeFileZipformerCtcFusion.java

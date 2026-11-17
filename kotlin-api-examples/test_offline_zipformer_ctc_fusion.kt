package com.k2fsa.sherpa.onnx

// This file shows how to fuse a KenLM language model into the search of a
// non-streaming Zipformer CTC model with decodingMethod prefix_beam_search.

fun main() {
  val config = OfflineRecognizerConfig(
      modelConfig = OfflineModelConfig(
          zipformerCtc = OfflineZipformerCtcModelConfig(
              model = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx",
          ),
          tokens = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt",
          numThreads = 1,
          debug = true,
      ),
      decodingMethod = "prefix_beam_search",
      // A character language model, fused with weight 0.35.
      fusion = FusionConfig(
          scorers = arrayOf(
              ScorerConfig(
                  type = "kenlm",
                  model = "./en-librispeech-char-6gram.bin",
                  weight = 0.35f,
              ),
          ),
      ),
      // Options the typed fields do not carry, by their command-line names;
      // this one adds a bonus per output token.
      optionsJson = """{"length-bonus": 0.5}""",
  )
  val recognizer = OfflineRecognizer(config = config)

  val waveFilename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav"
  val waveData = WaveReader.readWaveFromFile(
      filename = waveFilename,
  )

  val stream = recognizer.createStream()
  stream.acceptWaveform(waveData.samples, sampleRate = waveData.sampleRate)
  recognizer.decode(stream)

  println(recognizer.getResult(stream))

  stream.release()
  recognizer.release()
}

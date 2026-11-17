// Copyright 2026 graehl

// This file shows how to fuse a KenLM language model into the search of an
// offline Zipformer CTC model with decoding method prefix_beam_search.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileZipformerCtcFusion {
  public static void main(String[] args) {
    // please refer to
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
    // and
    // https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/en-librispeech-char-6gram.bin
    // to download model files
    String model = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx";
    String tokens = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt";
    String lm = "./en-librispeech-char-6gram.bin";

    String waveFilename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineZipformerCtcModelConfig zipformerCtc =
        OfflineZipformerCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setZipformerCtc(zipformerCtc)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    // A character language model, fused with weight 0.35.
    ScorerConfig scorer =
        ScorerConfig.builder().setType("kenlm").setModel(lm).setWeight(0.35f).build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("prefix_beam_search")
            .setFusion(FusionConfig.builder().setScorers(scorer).build())
            // Options the typed fields do not carry, by their command-line
            // names; this one adds a bonus per output token.
            .setOptionsJson("{\"length-bonus\": 0.5}")
            .build();

    OfflineRecognizer recognizer = new OfflineRecognizer(config);
    OfflineStream stream = recognizer.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    recognizer.decode(stream);

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}

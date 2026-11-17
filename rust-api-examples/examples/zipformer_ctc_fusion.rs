// rust-api-examples/examples/zipformer_ctc_fusion.rs
//
// Copyright (c)  2026  graehl
//
// Offline CTC decoding with prefix_beam_search and a KenLM language model
// fused into the search.
//
// clang-format off
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
// wget https://huggingface.co/graehl/sherpa-onnx-fusion-lm/resolve/9e61c0627e6cfc9ef62c355bcd62f796d97c26b4/en-librispeech-char-6gram.bin
//
// cargo run --release --example zipformer_ctc_fusion -- \
//   --model ./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx \
//   --tokens ./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt \
//   --lm ./en-librispeech-char-6gram.bin \
//   --wav ./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav
//
// clang-format on

use clap::Parser;
use sherpa_onnx::{
  OfflineRecognizer, OfflineRecognizerConfig, OfflineZipformerCtcModelConfig, ScorerConfig, Wave,
};

#[derive(Parser, Debug)]
struct Args {
  /// Path to the zipformer CTC model
  #[arg(long)]
  model: String,

  /// Path to tokens.txt
  #[arg(long)]
  tokens: String,

  /// Path to a KenLM model fused into the search
  #[arg(long)]
  lm: Option<String>,

  /// Search weight of the LM's log score
  #[arg(long, default_value_t = 0.6)]
  lm_weight: f32,

  /// JSON object of further options, e.g. '{"score-beam": 0}'
  #[arg(long)]
  options_json: Option<String>,

  /// Path to a wav file
  #[arg(long)]
  wav: String,

  /// Number of threads
  #[arg(long, default_value_t = 1)]
  num_threads: i32,
}

fn main() {
  let args = Args::parse();

  let mut config = OfflineRecognizerConfig::default();
  config.model_config.zipformer_ctc = OfflineZipformerCtcModelConfig {
    model: Some(args.model),
  };
  config.model_config.tokens = Some(args.tokens);
  config.model_config.num_threads = args.num_threads;
  config.decoding_method = Some("prefix_beam_search".into());

  if let Some(lm) = args.lm {
    config.fusion.scorers.push(ScorerConfig {
      r#type: Some("kenlm".into()),
      model: Some(lm),
      weight: args.lm_weight,
      ..Default::default()
    });
  }
  config.options_json = args.options_json;

  let recognizer =
    OfflineRecognizer::create(&config).expect("Failed to create OfflineRecognizer");

  let wave = Wave::read(&args.wav).expect("Failed to read WAV file");
  let stream = recognizer.create_stream();
  stream.accept_waveform(wave.sample_rate(), wave.samples());
  recognizer.decode(&stream);

  if let Some(result) = stream.get_result() {
    println!("Decoded text: {}", result.text);
  } else {
    eprintln!("No result produced");
  }
}

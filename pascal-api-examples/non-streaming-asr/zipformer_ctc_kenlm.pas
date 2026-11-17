{ Copyright (c)  2026  graehl }

{
This file shows how to fuse a KenLM language model into the prefix beam
search of a non-streaming Zipformer CTC model.

Please see ./run-zipformer-ctc-kenlm.sh for where to get the model files.
}

program zipformer_ctc_kenlm;

{$mode objfpc}

uses
  sherpa_onnx,
  SysUtils;

var
  Wave: TSherpaOnnxWave;
  Config: TSherpaOnnxOfflineRecognizerConfig;
  Recognizer: TSherpaOnnxOfflineRecognizer;
  Stream: TSherpaOnnxOfflineStream;
  RecognitionResult: TSherpaOnnxOfflineRecognizerResult;
begin
  Initialize(Config);

  Config.ModelConfig.ZipformerCtc.Model := './sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx';
  Config.ModelConfig.Tokens := './sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt';
  Config.ModelConfig.NumThreads := 2;

  Config.DecodingMethod := 'prefix_beam_search';
  Config.Fusion.CtcAlignment := 'max';

  { A character 6-gram over the LibriSpeech LM text. }
  SetLength(Config.Fusion.Scorers, 1);
  Config.Fusion.Scorers[0].&Type := 'kenlm';
  Config.Fusion.Scorers[0].Model := './en-librispeech-char-6gram.bin';
  Config.Fusion.Scorers[0].Weight := 0.35;

  { Any option that sherpa-onnx-offline --help lists can also be given as
    JSON; it is applied after the fields above. }
  Config.OptionsJson := '{"length-bonus-unit": "word"}';

  Recognizer := TSherpaOnnxOfflineRecognizer.Create(Config);
  if Recognizer.GetHandle = nil then
  begin
    WriteLn('Failed to create the recognizer. Please check your config');
    Halt(1);
  end;

  Wave := SherpaOnnxReadWave('./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/1.wav');
  Stream := Recognizer.CreateStream();
  Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);
  Recognizer.Decode(Stream);

  RecognitionResult := Recognizer.GetResult(Stream);
  WriteLn(RecognitionResult.Text);

  FreeAndNil(Stream);
  FreeAndNil(Recognizer);
end.

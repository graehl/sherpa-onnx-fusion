# LiteRT model backend

> Running a recognizer's model through LiteRT's `.tflite` form instead of
> ONNX Runtime, so a phone's accelerator (a Pixel's Tensor TPU) can run it;
> built for offline w2v-BERT 2.0 CTC models.

Topic: litert-backend

## What is built

- **Build.** `-DSHERPA_ONNX_ENABLE_LITERT=ON`, off by default;
  `cmake/litert.cmake` fetches the pinned SDK headers and, on Android,
  `libLiteRt.so` and the Tensor dispatch library.
- **Model.** `offline-w2v-bert-ctc-litert-model.{h,cc}` is chosen when
  `--w2v-bert-ctc` names a `.tflite` file. `--provider=npu` with
  `--litert-dispatch-dir` runs a compiled graph on the TPU; any other
  provider runs a portable graph on LiteRT's CPU kernels. The search and
  scorers stay outside LiteRT. Graphs from the research exporter since
  2026-11-24 end in a log-softmax, as k2-fsa's w2v-BERT export does (all
  of it on the TPU when compiled); the window blend normalizes again,
  which changes nothing for them and normalizes older graphs.
- **Fixed lengths.** A graph has one length per signature and takes
  `input_lengths`, so a shorter utterance is padded and masked.
  `--w2v-bert-ctc` takes several files, comma separated; an utterance runs
  in the shortest length that holds it.
- **Long audio.** A longer utterance runs in overlapping windows of one
  length (`--litert-window W:O:S`, seconds), blended per frame before one
  search, the same scheme as [chunked encoding](chunked-encoding.md),
  whose `--encoder.window.*` options it shares.
- **Bindings.** `litert_dispatch_dir` and `litert_window` are fields of
  the offline model config in every binding
  ([language bindings](language-bindings.md)).
- **A caller's environment.** By default the model creates and destroys
  its own LiteRT environment. `SherpaOnnxCreateOfflineRecognizerWithLiteRtEnvironment`
  (C), `OfflineRecognizer::Create(config, env)` (C++ wrapper) and
  `OfflineRecognizer::create_with_litert_environment` (Rust) build it in
  a `LiteRtEnvironment` the caller owns instead, so it shares that
  environment's device context with, say, an MT engine on LiteRT. The
  environment is borrowed and must outlive the recognizer;
  `litert_dispatch_dir` must then be empty, since the dispatch library is
  set when the environment is created. A null environment is the default.

Smoke-scale accuracy checks on one model, and phone timings, are recorded
in the [design sketch](../gaps/sketches/litert-model-backend.md#built).

## Design and remaining work

The [LiteRT sketch](../gaps/sketches/litert-model-backend.md) holds the
intended outcome (encoder first, the per-step models later if measurements
justify it), the design notes, the proposed Tensor backend path, and the
remaining steps in order: public test-set error rates against ONNX, the
model's description file, the Android app's packaging, and transducers
with the encoder in LiteRT.

---
slug: litert-model-backend
noticed: 2026-11-14
where: model execution backends, offline and online recognizers, Android builds
---

# Running models through LiteRT

## Intended outcome

User-directed (graehl, 2026-11-14): run recognizer models in LiteRT's
`.tflite` form, without going through ONNX, so that a phone's accelerator
can be used where LiteRT has a delegate for it and ONNX Runtime has no
execution provider. The interest is the Pixel's Tensor processor; the same
path would serve other phones LiteRT accelerates.

The first split is by cost and by what the search needs:

- **In LiteRT:** the encoder, up to its final hidden states, or up to the
  logits where the head is part of the same graph (a CTC model's output
  layer makes the graph depend on the vocabulary size).
- **Outside it at first:** everything after: a CTC model's log-softmax and
  the search; a transducer's prediction network and joiner; the scorers.
  These stay on the CPU or in ONNX Runtime as they are now.
- **Eventually, perhaps all of it** in LiteRT, if the small per-step models
  (joiner, prediction network, recurrent scorer) are worth moving.

## Why the encoder first

The encoder is nearly all of the acoustic model's compute and runs once per
utterance or chunk with no dependence on the search. The parts the search
calls per frame and per hypothesis are small, are called many times, and
have shapes that change with the beam, which accelerators and their
delegates handle poorly. The fork's search already takes encoder output as
plain arrays (`OfflineRecognizer::EncodeStreams` for CTC; the transducer
decoders take the encoder tensor), so the encoder can come from another
runtime without the search knowing.

## Design notes

- **Where it fits.** sherpa-onnx already has per-backend model classes
  beside the ONNX ones (RKNN, QNN, Axera, Ascend), selected by provider
  and model configuration. A LiteRT backend is another of those: a model
  class that produces the same encoder output, behind the same recognizer
  implementations, built only when a CMake option is on.
- **Fixed shapes.** Accelerator delegates generally want static input
  shapes. A whole-utterance encoder then needs either a set of exported
  lengths with padding to the next one, or chunked processing. Padding
  brings the masking question in
  [batch changes transcripts](../offline-batch-changes-transcripts.md):
  the export has to take a length and mask with it, or the padded result
  differs from the unpadded one.
- **Quantization.** Accelerators generally require integer models with
  fixed (static) ranges. That is a different thing from the dynamic int8
  quantization measured so far, and it would remove the dependence of one
  utterance's output on its batch neighbours. Its accuracy has to be
  measured per model.
- **Conversion belongs with the model's export**, in the repository that
  trains it. This repository takes the `.tflite` file and its small
  description (feature options, output kind, frame rate, token table).
- **The Tensor path exists.** LiteRT 2.3.0 reaches the Pixel 10 and 11
  TPU: a graph is compiled ahead of time for the chip (Tensor G5 or G6)
  on the host, because the phone cannot compile, and run on the phone
  through the dispatch library in the Google Tensor NPU runtime AAR.
  Without an accelerator, LiteRT on the CPU (XNNPACK) is only an
  alternative to ONNX Runtime on the CPU, to be justified by a
  measurement.
- **Operators.** The encoders use attention with relative position terms,
  layer normalization variants, and convolutions; a delegate that lacks one
  operator sends that part of the graph back to the CPU and can lose the
  gain. Operator coverage is checked per model with the delegate's own
  report, not assumed.

## Built

For offline w2v-BERT 2.0 CTC models (2026-11-15):
`-DSHERPA_ONNX_ENABLE_LITERT=ON` (off by default; `cmake/litert.cmake`
fetches the pinned 2.3.0 SDK headers and, on Android, `libLiteRt.so` and
the Tensor dispatch library) and `offline-w2v-bert-ctc-litert-model.{h,cc}`,
chosen when `--w2v-bert-ctc` names a `.tflite` file. `--provider=npu` with
`--litert-dispatch-dir` runs a compiled graph on the TPU; any other
provider runs a portable graph on LiteRT's CPU kernels. The graph has one
fixed length P and takes `input_lengths`, so an utterance is zero padded
to P and masked; a longer one is cut to P with an error logged.

Checked: on a Pixel 11 a compiled 5 s graph ran entirely on the TPU
through `sherpa-onnx-offline`; on an x86 host the portable graph, linked
against the `ai-edge-litert` wheel's `libLiteRt.so`, gave the ONNX fp32
graph's transcripts. The compiler's default reduced precision on the TPU
changes a few characters of some transcripts.

## Remaining, in order

1. **Longer audio, and short audio cheaply.** On a Pixel 11, 10 s and
   20 s graphs of a 600M-parameter model load and run; a 30 s graph
   compiled but failed to load: the TPU runtime could not allocate its
   DMA-BUF memory ("Failed to allocate from DMA-BUF heap system: Out of
   memory"). A graph costs the same whatever the utterance's length, so a
   short utterance in a long graph can be slower than the CPU. Either load
   several lengths and run each utterance in the shortest that holds it
   (their memory together is unmeasured), or run fixed-length windows,
   which needs overlap and stitching for a CTC model.
2. **Error rate**, not single transcripts, of the TPU path against ONNX
   fp32 and int8 on a test set, and whether the compiler's full-precision
   option is worth its cost.
3. **The model's description file.** The graph's length and frame rate
   come from its signature now; the feature options are still the
   w2v-BERT ones the recognizer sets.
4. Bindings (C API, JNI/Kotlin) for `litert_dispatch_dir`, and the
   Android app's packaging of the two libraries.
5. Transducers: encoder in LiteRT, prediction network and joiner in ONNX
   Runtime.
6. Decide from measurements whether the per-step models should move.

## Completion evidence

On a phone with an accelerator delegate: the same transcripts, within a
stated tolerance, as the ONNX path on a test set; encoder time and energy
against the ONNX CPU path; and the fraction of the encoder the delegate
ran.

Contributing-model: fable-5.2

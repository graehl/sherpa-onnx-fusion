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
fixed length P and takes `input_lengths`, so a shorter utterance is zero
padded to P and masked. Longer audio uses the windows described below.

Several lengths and long audio (2026-11-16). A graph costs the same
whatever the utterance's length (on a Pixel 11, a 600M-parameter model's
5, 10 and 20 s graphs take about 0.10, 0.30 and 1.5 s per call), and a
30 s graph of that model did not load (the TPU runtime could not allocate
its DMA-BUF memory). So `--w2v-bert-ctc` takes several `.tflite` files,
comma separated, and each signature of each is one length; a file of
several signatures holds the weights once (about 0.7 GB per extra length
instead of 1.3 for that model). An utterance runs in the shortest length
that holds it. A longer one runs in overlapping windows of one length
(`--litert-window W:O:S`, seconds; default the longest length, 3 s
overlap, 1 s slack): evenly spaced, interior windows moved to put their
edges where the filterbank energy is lowest, each window's features
normalized over the window, logits blended per frame with weight
1 + distance from the window's nearer edge, then one search over the
whole utterance: center-weighted stitching of overlapping chunk encodings,
as used for long inputs to bidirectional text encoders.

Checked: on a Pixel 11 a compiled 5 s graph ran entirely on the TPU
through `sherpa-onnx-offline`; on an x86 host the portable graph, linked
against the `ai-edge-litert` wheel's `libLiteRt.so`, gave the ONNX fp32
graph's transcripts, and a 20 s utterance in three 10 s windows gave the
same greedy transcript as an independent implementation of the scheme.
On the Pixel 11 TPU, with 5 and 10 s bf16 graphs and
`--litert-window=10:3:1`, greedy decoding of 2, 5, 10, 20 and 30 s clips
took 0.13, 0.10, 0.34, 1.06 and 1.41 s warm (RTF 0.046), the 30 s clip
whole. The compiler's default precision (bf16) changes a few characters
of some transcripts.

## Remaining, in order

1. **Error rate in this repository**, not single transcripts: the TPU
   path, with lengths and windows, against ONNX fp32 and int8 on a public
   test set, as `repro/` reruns other claims.
2. **The model's description file.** The graph's length and frame rate
   come from its signature; supported outputs have one frame per input
   row or per two rows. The feature options are still the
   w2v-BERT ones the recognizer sets.
3. Bindings (C API, JNI/Kotlin) for `litert_dispatch_dir` and
   `litert_window`, and the Android app's packaging of the two libraries.
4. Transducers: encoder in LiteRT, prediction network and joiner in ONNX
   Runtime.
5. Decide from measurements whether the per-step models should move.

## Completion evidence

On a phone with an accelerator delegate: the same transcripts, within a
stated tolerance, as the ONNX path on a test set; encoder time and energy
against the ONNX CPU path; and the fraction of the encoder the delegate
ran.

Contributing-model: fable-5.2

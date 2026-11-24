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
  layer makes the graph depend on the vocabulary size), and since
  2026-11-24 a CTC model's log-softmax, which compiles onto the TPU.
- **Outside it at first:** everything after: the search; a transducer's
  prediction network and joiner; the scorers.
  These stay on the CPU or in ONNX Runtime as they are now.
- **Eventually, perhaps all of it** in LiteRT, if the small per-step models
  (joiner, prediction network, recurrent scorer) are worth moving.

## Why the encoder first

The encoder is nearly all of the acoustic model's compute and runs once per
utterance or chunk with no dependence on the search. The parts the search
calls per frame and per hypothesis are small, are called many times, and
have shapes that change with the beam, which accelerators and their
delegates handle poorly. The fork's search already takes encoder output as
plain arrays (`OfflineRecognizer::EncodeStreams` leaves them on the stream
for CTC models and the zipformer and NeMo transducers), so the encoder can
come from another runtime without the search knowing.

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
- **Precision.** Integer quantization is not a universal accelerator
  requirement: the Tensor compiler supports floating-point truncation,
  including bf16 and half. Static integer quantization, where supported,
  avoids batch-dependent dynamic activation ranges; it does not fix
  masking or normalization differences. Measure accuracy per model and
  compilation mode ([Tensor compilation flags](https://developers.google.com/edge/tensor-sdk/compilation-flags)).
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

Smoke-scale accuracy checks exist for one w2v-BERT 2.0 CTC model:
on a Pixel 11 a compiled 5 s graph ran entirely on the TPU
through `sherpa-onnx-offline`; on an x86 host the portable graph, linked
against the `ai-edge-litert` wheel's `libLiteRt.so`, gave the ONNX fp32
graph's transcripts, and a 20 s utterance in three 10 s windows gave the
same greedy transcript as an independent implementation of the scheme.
On the Pixel 11 TPU, with 5 and 10 s bf16 graphs and
`--litert-window=10:3:1`, greedy decoding of 2, 5, 10, 20 and 30 s clips
took 0.13, 0.10, 0.34, 1.06 and 1.41 s warm (RTF 0.046), the 30 s clip
whole. The compiler's default precision (bf16) changes a few characters
of some transcripts.

## Proposed shared tiling and Tensor backend path

User-directed (graehl, 2026-11-16): make center-embed tiling of fixed audio
length graphs available independently of the LiteRT build, for NPU/GPU
backends generally; sketch the path from encoder acceleration to possible
LM acceleration.

### Tiling belongs above the runtime: built

Built 2026-11-16 as [chunked encoding](../../topics/chunked-encoding.md):
the scheduler and the join are built unconditionally, LiteRT uses them,
and ONNX Runtime runs every CTC model, the zipformer transducer and NeMo
RNN-T/TDT in windows (`--encoder.window.*`), with input lengths chosen
from a set (`--encoder.lengths`, `--encoder.pad-to`) as LiteRT's graphs
are. Normalization scope, weighting and the blend space are options;
`--litert-window` is kept. What remains of this part is the comparison
of settings, and against whole-utterance inference and the LiteRT path,
on a test set of long recordings.

### A sherpa backend is practical; an ONNX Runtime EP is a larger project

The existing w2v-BERT implementation establishes feasibility for a
selected encoder family. Extend that sherpa-level backend first, using
`OfflineCtcModel` and model-family dispatch rather than replacing every
use of `Ort::Value`. Keep LiteRT optional, as KenLM is; common tiling
must neither include LiteRT headers nor fetch or link its libraries.

An actual ONNX Runtime execution provider (EP) claims ONNX nodes or
subgraphs through `GetCapability()`. The
[published EP list](https://onnxruntime.ai/docs/execution-providers/)
does not list a Google Tensor/LiteRT provider as of 2026-11-16.
A future provider is technically plausible, but SDK beta exit alone
does not provide ONNX lowering, shape restrictions, compilation caches,
device/runtime compatibility, or a maintainer for that integration.
The practical prospect is stronger for selected supported model families
than for arbitrary ONNX graphs; there is no basis here for a release date
or probability estimate for an upstream EP.

The current [Tensor SDK compiler](https://developers.google.com/edge/tensor-sdk/tensor-faqs)
requires a valid `.tflite` model with static shapes. Google's
[LiteRT NPU documentation](https://developers.google.com/edge/litert/next/npu)
specifies ahead-of-time compilation for Tensor and no on-device JIT in
the beta. Use source-model export to LiteRT, followed by chip-specific
compilation, as the supported route for each selected family. `.tflite`
is necessary for that compiler today; it need not become sherpa's
universal model format.

[onnx2tf](https://pypi.org/project/onnx2tf/2.4.0/) already converts ONNX
models to LiteRT/TFLite. Producing a flatbuffer is only one gate: the
converted graph must preserve model semantics, compile for Tensor, fit
device memory, and run efficiently. Treat ONNX conversion as a verified
build-time recipe per family. Prefer direct source-framework exports
when they avoid conversion problems; do not wait for an arbitrary-ONNX
converter before extending this backend.

Revisit an ORT EP when a maintained Tensor/LiteRT provider or reusable
ONNX lowering path exists. A narrower custom EP could recognize known
encoder subgraphs and load matching precompiled artifacts, but would
still need artifact matching and fallback rules. Until that buys more
than the existing sherpa model dispatch, it adds integration work without
expanding supported models.

### Extend acceleration by measured stage cost

Keep acoustic encoding, search, and scorer execution separately selectable:

```text
features -> shared length selection / tiling
         -> encoder adapter: ORT | LiteRT | native backend
         -> aligned outputs -> CTC or transducer search on CPU
                            -> predictor / joiner / neural LM adapters
```

For transducers, retain the predictor and joiner in ONNX Runtime first.
For neural LM fusion, `RecurrentLmStep` already separates a batched
token/state computation from prefix-state management. A LiteRT step
adapter could batch active hypotheses into fixed-size buckets, provided
it preserves state ownership and score semantics. Beam expansion,
prefix tries, and KenLM remain CPU work. Moving a neural LM step to the
TPU does not imply that the whole fusion algorithm should run there.

Measure each candidate with dispatch, padding, synchronization, state
transfer, and cache reuse included. Small sequential steps may lose to
CPU execution; shared device buffers need explicit ownership and lifetime
rules before any zero-copy claim. Report which stages ran on which
device, and make whole-stage fallback explicit so a successful decode
cannot conceal failed acceleration.

Package each supported export with a manifest recording model and
artifact hashes, feature normalization, vocabulary, tensor names and
dtypes, lengths/masks, output cadence, and compiler/runtime/chip versions.
Portable CPU and compiled TPU artifacts must refer to the same model
contract. Validate numerical agreement and word error rate alongside
latency, peak memory, and accelerator coverage.

Contributing-model: 6-Astra

## Remaining, in order

1. **Broaden the existing smoke-scale accuracy checks** above to public
   test-set error rates: the TPU path, with lengths and windows, against
   ONNX fp32 and int8, as `repro/` reruns other claims. The existing
   transcript comparisons cover one model; broader accuracy evaluation
   remains to be done.
2. **The model's description file.** The graph's length and frame rate
   come from its signature; supported outputs have one frame per input
   row or per two rows. The feature options are still the
   w2v-BERT ones the recognizer sets.
3. The Android app's packaging of the two libraries. (The options are in
   every binding: [language bindings](../../topics/language-bindings.md).)
4. Transducers: encoder in LiteRT, prediction network and joiner in ONNX
   Runtime.
5. Decide from measurements whether the per-step models should move.

## Completion evidence

On a phone with an accelerator delegate: the same transcripts, within a
stated tolerance, as the ONNX path on a test set; encoder time and energy
against the ONNX CPU path; and the fraction of the encoder the delegate
ran.

Contributing-model: fable-5.2

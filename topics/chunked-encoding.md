# Chunked encoding of long utterances

> An offline recognizer can encode an utterance longer than a set length
> in overlapping windows, batched like utterances, and blend the windows'
> outputs per output frame before one search over the whole utterance;
> and it can pad every encoder input to one of a few lengths so that an
> accelerator sees few distinct shapes.

Topic: `chunked-encoding`

## Scope and status

Built for every offline model family the fork's search runs on: the CTC
models (zipformer, NeMo, WeNet, w2v-BERT, Dolphin, MedASR, FireRed,
TeleSpeech, TDNN), the zipformer transducer, and NeMo RNN-T and TDT. Not
for an omnilingual CTC model, whose input is raw samples (the option is
an error there), nor for the attention encoder-decoder families (Whisper,
Moonshine, Canary and the like), whose decoders attend to every encoder
frame. A LiteRT `.tflite` model windows itself with the same scheduler
and blend (see [LiteRT](#litert-models)).

The generic encoder (the shared path for the models other than LiteRT)
is off by default: with no `--encoder.*` option set, every utterance
is encoded whole, as upstream does. With windows on, an utterance no
longer than the window is still encoded whole. With no padding options,
in a call holding only such utterances it gets exactly the output it gets
with the option off;
in a call that also holds a windowed one it is padded to the window
length instead of to the longest utterance, which can change its
transcript for an export whose results depend on the batch
([batch changes transcripts](../gaps/offline-batch-changes-transcripts.md)).

The options are on the C++ configuration and the command-line programs
only, like the fork's other options.

## What it is for

- **Memory and time on long audio.** These encoders attend over the whole
  utterance, so their cost grows faster than its length. Windows bound
  the sequence length of every call.
- **Bounded, few shapes for accelerators.** A GPU or NPU run through ONNX
  Runtime pays when the input length changes (new kernels or a
  recompiled graph), and a fixed-shape LiteRT graph cannot take any other
  length at all.

What it costs: overlap is computed twice, and each window sees only its
own context, so the result approximates the whole-utterance encoding;
it is not equal to it. Stateful streaming models are a different case:
they have a trained chunk-and-cache protocol and do not use this.

## Options

| option | default | meaning |
|---|---|---|
| `--encoder.window.seconds` | 0 | window length; 0 encodes every utterance whole |
| `--encoder.window.overlap` | 3 | overlap used to space windows before grid alignment and slack adjustment, seconds |
| `--encoder.window.slack` | 1 | how far an interior window may move to put its edges in quiet audio; 0 keeps windows evenly spaced |
| `--encoder.window.normalize` | `window` | `window` or `utterance`: the frames over which a feature normalization that uses its input's statistics is computed |
| `--encoder.window.weight` | `center` | `center` averages the windows covering a frame, each weighted by 1 + the frame's distance from that window's nearer edge; `cut` takes the frame from the window in which it is farthest from an edge |
| `--encoder.window.blend` | `log` | CTC only: `log` averages log probabilities and renormalizes; `prob` averages probabilities |
| `--encoder.lengths` | empty | comma-separated seconds: pad every encoder input to the shortest of these that holds it |
| `--encoder.pad-to` | 0 | pad an input that `--encoder.lengths` does not hold to a multiple of this many seconds |
| `--litert-window` | empty | `W:O:S`, LiteRT only, kept for earlier runs: overrides window seconds, overlap and slack |

`--encoder.lengths` and `--encoder.pad-to` need an encoder that masks
padding past the longest input of a batch; they are an error, when the
model loads, for one that takes no padded batch
(`SupportBatchProcessing` false), for the zipformer CTC and transducer
exports, which size their mask by the longest of the lengths and fail
in ONNX Runtime when padded further, and for TDNN, which reports every
padded frame as valid. The NeMo TDT export and a w2v-BERT export that
takes `input_lengths` were checked: padded to 120 and 30 s they ran, the
w2v-BERT transcript unchanged. They apply to whole utterances as well
as to windows, so they are useful without windows. With either, the
window becomes the shortest allowed length that holds
`--encoder.window.seconds`, so that every full window runs at one shape;
the last window, which alignment can leave up to one output frame
short, is padded to it. An input longer than every listed length falls
to `--encoder.pad-to`, else runs at its own length.

Example: 20 s windows, with every call at 10 or 20 s or a multiple of 5 s:

```text
--encoder.window.seconds=20 --encoder.lengths=10,20 --encoder.pad-to=5
```

## Where windows go and how they are joined

Window starts are multiples of the model's subsampling factor, so that
output frame `j` of a window starting at input frame `s` is output frame
`s / factor + j` of the utterance. The schedule takes as few windows as
keep neighbours overlapping by at least `overlap`, spaced evenly, the
first at the start and the last ending at the end. With `slack`, each
interior window then moves by up to that much to the aligned start
whose two edge frames are quietest (lowest summed filterbank values),
keeping half the minimum overlap with each neighbour. For the generic
encoder, alignment and movement also preserve at least four output
frames of overlap to cover the permitted front-end edge loss.

All windows of the utterances in one call, and the whole utterances
that need none, go to the encoder in one padded batch when the model
takes one, else one call each. Windows of one length need almost no
padding. Then each output frame is joined from the windows that cover
it, by `weight` and `blend`, and the search runs once over the joined
utterance, so timestamps are on the utterance's own frame grid. Every
output frame must be covered. For the generic encoder, an overlap shorter
than four output frames is an error when the model loads.

What is joined depends on the family. A CTC model's output is a score
per token; with `blend=log` the windows' log probabilities are
averaged and renormalized, which after the softmax is the same as
averaging logits. A transducer's encoder output is hidden states,
averaged as they are; the joiner projects them linearly first, so this
is close to averaging the joiner's encoder term. `blend=prob` is an
error for a transducer.

## Choices that are open

The generic encoder's settings have not been compared on a test set.
The defaults reproduce the scheme the LiteRT backend first used;
the Lao comparison below covers that backend on one model. Each of
these choices remains open across model families:

- **Normalization scope.** It matters only for a model that normalizes
  features with statistics of its own input: w2v-BERT's per-utterance
  mean and variance, NeMo's `per_feature`. Per window, each window looks
  like an utterance of its own, as at training; per utterance, the
  statistics are steadier and the same in every window, so windows
  agree better where they overlap. Fixed-statistics normalizations
  (Dolphin, FireRed) and zipformer's none are unaffected.
- **Blend space.** `log` gives a renormalized geometric mean of the
  windows' distributions, which favours tokens every window finds
  likely; `prob` gives the arithmetic mean, a mixture, which keeps a
  token one confident window found.
- **Weight.** `center` smooths the seam; `cut` makes each frame one
  window's own output and never mixes hidden states.
- **Overlap and slack.** More overlap gives each frame more context at
  more compute; slack moves seams into pauses.

## LiteRT models

A `.tflite` model has a fixed set of lengths, one per signature, and
windows itself: an utterance runs in the shortest length that holds it,
a longer one in windows of the shortest length of at least
`--encoder.window.seconds` (0: its longest), with the shared scheduler
and blend and `--encoder.window.normalize`, `weight` and `blend`.
`--litert-window W:O:S` overrides the window, overlap and slack; the
padding options do not apply. The recorded LiteRT runs (the LiteRT
sketch's Pixel 11 timings with 5 and 10 s graphs, and its x86 check of a
20 s utterance in 10 s windows) used `--litert-window=10:3:1` before the
other options existed; that, or `--encoder.window.seconds=10` with the
default overlap, slack, normalize, weight and blend, runs them as they
ran.

## Encoding apart from the search

Chunked encoding sits in the step that runs the encoder ahead of the
search, `OfflineRecognizer::EncodeStreams`, which the batched masked
encoding of w2v-BERT and omnilingual exports and
`sherpa-onnx-offline-parallel --decode-threads` already used for CTC.
The zipformer and NeMo transducers have that step too, so for every
family above the model threads only encode and the search threads only
search, with windows or without; transcripts are the same either way.
`DecodeStreams` alone also works: it encodes what is not yet encoded,
and with no `--encoder.*` option set and nothing encoded ahead it takes
upstream's own path (`OfflineStreamEncoder::Active`).

## For developers

The facility is two pieces, neither depending on a runtime:

- `offline-encoder-window.{h,cc}`: the options (`OfflineEncoderConfig`,
  `OfflineEncoderWindowConfig`), the schedule (`EvenWindowStarts`, in
  whatever input unit the caller uses) and the join (`WindowBlend`).
  LiteRT uses these directly, in its input rows.
- `offline-stream-encoder.{h,cc}`: `OfflineStreamEncoder` encodes a set
  of `OfflineStream`s, whole or in windows, batched as the model allows
  and padded to the allowed lengths, and keeps each stream's valid
  output frames on it; `TakeEncoderOutputs` re-pads those into the
  batch tensor a decoder takes.

A model family plugs in with an `OfflineEncoderAdapter`: a function that
runs its encoder on a padded batch and returns time-major output with
valid lengths, its feature normalization, subsampling factor, padding
value, whether it takes a batch, and whether its output is scores or
hidden states. The CTC, zipformer-transducer and NeMo-transducer
recognizers each build one in a few lines (`InitEncoder`); a new family
with a separable encoder needs the same and nothing else here.

Normalization is the adapter's `normalize`: a CTC model's
`NormalizeFeatures`. NeMo's `per_feature` is otherwise done by the
stream over the whole utterance; with windows on, the recognizer calls
`MoveStreamNormalizationToEncoder`, which turns it off in the stream
(`FeatureExtractorConfig::nemo_normalize_in_stream`) and puts it first in
the adapter's `normalize`, so that both scopes apply to it and window
placement sees the same unnormalized features under either. The
normalize type itself must stay set: it also selects NeMo's log guard
in the filterbank, and clearing it changed transcripts.

A window's output frame count is checked against the subsampling
factor (within four output frames, for front-end edge loss); a model
whose `SubsamplingFactor` is wrong fails loudly instead of misaligning
the join.

A w2v-BERT ONNX export with strided adapter layers must set
`subsampling_factor` metadata to its total stride from 10 ms filterbank
frames, including the factor of two from feature stacking. Without that
metadata, the runtime assumes two. The value controls window alignment
and timestamps; an export with one adapter of stride two needs four.

## Evidence

`offline-encoder-window-test.cc` checks the schedule's coverage and
alignment, each blend mode's arithmetic, and, through real
`OfflineStream`s with a stand-in encoder that is local in time, that
windows reproduce the whole-utterance output with and without batching,
that normalization scope changes the output only for long utterances,
and that inputs reach the encoder at the allowed lengths.

A smoke run (2026-11-16, two threads, one call per model holding a
5.9 s LibriSpeech test-other utterance and two concatenations of 46 and
92 s): windows of 200 s, longer than everything, gave transcripts
identical to the option off for all three public models below. With
20 s windows the 92 s file's words differed from whole-utterance
decoding in 4 to 5 places of about 200, and peak memory and time fell:

| model | peak memory, whole → 20 s windows | time |
|---|---|---|
| zipformer CTC (en, int8) | 7.1 → 1.9 GB | 9.0 → 3.2 s |
| zipformer transducer (small en, int8) | 6.2 → 1.6 GB | 8.2 → 2.9 s |
| Parakeet TDT 110M (int8) | 2.4 → 1.5 GB | 9.2 → 4.5 s |

Two long files say nothing about which setting is more accurate; that
needs a test set of long recordings.

For LiteRT, on an x86 host with portable 10 and 20 s graphs of a
w2v-BERT CTC model and the same three files, the binary before the
shared facility and the one after gave byte-identical results by
default and with `--litert-window=10:3:1`, and
`--encoder.window.seconds=10` gave the same bytes as that alias.

### Lao quality and real-time factor

Measurements from 2026-11-15/16 on 405 FLEURS Lao test clips, using the
same private 600M w2v-BERT CTC student. ER2 is the promoted Lao
character-based error metric used for this evaluation, not plain WER;
lower is better. The model and scoring recipe are not published here,
so these figures are deployment evidence, not a public reproduction
recipe. The fused
decode uses a character 8-gram, beam 32, LM weight 0.3 and bonus 1.
These are batch-1 results from the standalone phone driver, not a
quality evaluation of the fork's generic encoder.

| Encoding path | Greedy ER2 | With 8-gram ER2 | Warm fixture RTF, with 8-gram | Test-pass RTF, with 8-gram |
|---|---:|---:|---:|---:|
| Phone ONNX CPU, int8, 20 s pause pieces | 14.89 | 12.34 | — | — |
| Pixel 11 LiteRT TPU, bf16, 20 s pause pieces | 14.43 | 12.07 | 0.091 | 0.106 |
| Pixel 11 LiteRT TPU, bf16, overlapping 10 s windows | 14.52 | 12.01 | 0.062 | 0.075 |
| Pixel 11 LiteRT TPU, bf16, overlapping 5 s windows | 15.20 | 12.39 | 0.051 | 0.062 |
| Pixel 11 LiteRT TPU, int8 feed-forward layers, 10 s windows | — | 12.11 | 0.085 | 0.092 |
| Pixel 11 LiteRT TPU, int8 feed-forward layers, 5 s windows | 15.23 | 12.47 | — | — |
| Host PyTorch bf16, whole clips | — | 11.93 | — | — |

RTF is decode seconds divided by audio seconds, including features,
encoding and search. Warm fixture timing uses 2/5/10/20/30 s Lao clips
(67 s total), one warm-up and the median of three runs, USB powered,
after at least 60 s idle and with skin temperature at most 28 °C.
Test-pass timing has no warm-up per clip. Dashes mean no matching
timing is supplied here; the CPU quality row was run on an S22+, so it
does not provide a same-device speed comparison with the Pixel 11.

The pause-piece baseline cuts at pauses with a 20 s target and searches
each piece separately. It uses 5, 10 and 20 s graphs, choosing the
shortest that holds each piece. The 10 s windows use 5 and 10 s graphs,
3 s overlap and 1 s slack; the 5 s windows use a 5 s graph, 2 s overlap
and 1 s slack. Both normalize per window and blend with center weights.

The bf16 TPU 20 s pause-piece row scores 0.27 points below int8 ONNX
with the LM (paired
bootstrap, 10,000 resamples; 95% interval for TPU minus ONNX:
−0.44 to −0.11; Holm-adjusted p = 0.002). Precision and execution paths
differ, so this does not isolate a runtime effect. Against 20 s pieces,
10 s windows change ER2 by −0.06 (95% interval −0.21 to +0.09), while
5 s windows change it by +0.31 (+0.00 to +0.62); neither window
comparison is significant after the recorded multiple-comparison
correction. These data do not establish equivalence.

Batch 2 has executed fully on the Pixel TPU, but a matched batch-1
versus batch-2 quality comparison remains unmeasured. It needs identical
features, physical lengths, windows, stitching and decoding on both
widths, with a float ONNX reference. The table does not measure a
batching penalty or the isolated benefit of silence anchoring.

The int8 TPU rows quantize weights and activations in the feed-forward
layers (about two thirds of the weights), calibrated on 32 FLEURS Lao
validation clips; the rest stays bf16. This is mixed precision, not a
fully int8 encoder. At 10 s it changes ER2 by +0.10 against bf16
(95% interval −0.04 to +0.23) and is slower. At 5 s it changes ER2 by
+0.08 (−0.06 to +0.22); the table-free encoder call takes 0.090 s
against 0.110 s for table-free bf16, but that is not an end-to-end RTF.
Table-free graphs compute the position term rather than storing its
table. The 5 s encoder speedup does not imply an end-to-end speedup at
10 s. Weight-only
int8 was also tried: its 5 s encoder call took 0.603 s, slower than
bf16, because weights were dequantized on each call.

## Not done

- A word error rate comparison of the settings on long recordings.
- Bindings for the options.
- Starting the search on finished windows before the whole utterance is
  encoded: [hybrid chunked streaming](../gaps/sketches/hybrid-chunked-streaming.md).

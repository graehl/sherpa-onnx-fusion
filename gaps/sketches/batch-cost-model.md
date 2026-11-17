---
slug: batch-cost-model
noticed: 2026-11-15
where: sherpa-onnx-offline-parallel (SplitToSimilarBatches, BatchingCost)
---

# A measured cost model for a batch

## Intended outcome

User-directed (graehl, 2026-11-15): model the time of one batch as a
function of its width (number of recordings) and length (its longest
recording), and cut the sorted list by that.

`sherpa-onnx-offline-parallel` cuts a length-sorted list into batches with
a dynamic program whose cost for a batch is

    overhead + width * length

in seconds of audio, with `--batch-overhead-seconds` for the overhead. The
default overhead is so large that it means "fewest batches". Nothing has
been fitted to a timing, so the cut is exact for a model nobody has
checked. The dynamic program costs each candidate span on its own, so any
function of width and length can replace the formula without changing the
search.

## What a fitted model might need

- **A real overhead.** On CUDA with the zipformer CTC model a call cost
  about 13 ms at a repeated input shape and about 60 ms at a new one,
  against roughly 0.3 ms per second of padded audio
  ([GPU timing](../gpu-decode-cost-outside-the-model.md)). That puts the
  overhead between about 35 and 200 seconds of audio depending on whether
  shapes repeat, so the per-shape cost has to be settled first. Before
  shape enters the model, learn what is cached and for how long (graehl,
  2026-11-15). In the measured process the cost recurred on every
  change of shape from the previous call, seen shape or not. So the term
  to model is "this batch's shape differs from the last one's", which
  depends on the order of batches and on rounding lengths up to shared
  sizes, not on the set of shapes. Caching across processes is untested.
- **Deliberate extra padding.** If a change of shape costs more than the
  padding that avoids it, the best cut pads batches up to a shared length
  and keeps their widths equal, so that consecutive batches have one
  shape. Rough break-even from the numbers above: a change costs about as
  much as 150 s of padded audio at width 1 and perhaps 35 s at width 16,
  the latter from two calls. In a large sorted list consecutive batches
  differ by a fraction of a second, so rounding up to a coarse grid would
  cost little padding and save the change on most calls. The search then
  has to carry the previous batch's shape, which stays small if lengths
  are on a grid. A change of width alone costs as much as a change of
  length (measured 2026-11-15), so rounding lengths up at chosen cut
  points is not enough: the unit to choose is a run of consecutive
  batches with one width and one padded length (graehl, 2026-11-15). A
  run over a span of the sorted list costs one change plus its batches at
  that shape, and its span must divide into that width, with the
  leftover a narrower run of its own. For a model that does not mask padding this also changes
  more transcripts.
- **Time that is not linear in width.** A GPU that is not saturated by a
  narrow batch makes extra width nearly free up to some point. The formula
  charges every recording the same.
- **Preferred widths.** Some multiple of width, say 4 or 8, may run
  slightly faster than its neighbours. graehl doubts it matters; check it
  only as a by-product of the grid below.
- **A memory limit that is not width times length.** `--max-batch-seconds`
  assumes memory follows padded audio. Attention over time grows faster
  than linearly in length.
- **Search time.** With the prefix search and a scorer the search is a
  larger share and has its own batching, so the encoder model alone will
  overstate the gain.

## Windowed inputs and one packer

User-directed (graehl, 2026-11-17): use one batch packer for whole
utterances and encoder windows, including the parallel offline driver.
Form windows before sorting and packing them. Keep each window's source
utterance and output offset independently of its execution order.

The physical slot is one row of the encoder input batch, whose time axis
is rounded to an allowed length (a bucket). When the source is at least
that long, fill the slot with real context. Rounding an 11 s window to a
12 s bucket should encode 12 s of audio. This length check is the difference from packing a
whole utterance; it does not justify a separate packer. For exports without
`input_lengths` or `x_lens`, fully occupied slots are the preferred default,
except for very short utterances. Batchability and padding-mask support
must be separate capabilities.

Overlap permits some movement toward silence without changing physical
length. Bound that movement so window centers do not move too far; full
occupancy takes priority over silence anchoring. The benefit of silence
anchoring remains unmeasured. LiteRT shapes must come from the supplied
prebuilt graphs, including their batch widths. Batch width one remains the
single-user default; larger widths are intended for serving too.

This is follow-up design, not a defect in the initial chunked-encoding
contract. The current generic encoder sends all expanded windows from one
utterance batch in one call; it does not yet bound or repack that expansion.

## Confident sections before overlapping windows

User-directed (graehl, 2026-11-17): a sufficiently long, confident silence
may first divide a long recording into independent sections. Those are
hard boundaries: no encoder window may cross them. Apply overlapping
windows only within a section that is long enough to need them, keeping
original recording offsets for results and timestamps. Define the silence
confidence and minimum duration separately from the encoder length cap.

The no-overlap baseline solves a different segmentation problem: each
piece must fit its length cap, so it chooses a quiet boundary when it can
and makes a hard cut even without silence when necessary. It should not
be represented by fixed no-overlap tiling. A comparison need not force
identical segmentation on methods with different constraints; it must
state those constraints and score the same complete recordings.

Overlap need not make middle frames run twice. The existing
`--encoder.window.overlap` option sets an absolute duration independently
of the window length; smaller values can reduce repeated computation
on long inputs, subject to the current four-output-frame coverage floor.
An automatic policy that varies overlap with recording or section length
would be additional work. Measure quality first, and total compute before
claiming that overlap is as efficient as hard splitting.

Overlap and weighting are separate decisions. Equal weights across all
covering windows give uniform averaging; center averaging instead weights
each frame by its distance from the window edge. The latter is believed
to be better with suitable parameters, not established as universally
better. Compare both on the same windows before attributing a gain to
center weights. The generic encoder currently exposes `center` and `cut`,
not uniform averaging; `cut` selects one window's frame and is not the
uniform-average condition.

## How to measure

### Diarization-guided window placement

User-directed (graehl, 2026-11-17): optionally use an actual diarization
model that can process long recordings to choose encoding splits and
overlaps, instead of silence guesses or in addition to them. The aim is
better recognition even when the application does not need annotations
of who speaks when. Better WER is a hypothesis to test.

The diarizer's speaker-change and speech-activity predictions can supply
boundary scores to the window scheduler. Keep that proposal separate
from splitting the decoder state at every speaker change: window placement
alone can preserve one search over the joined recording. Speaker changes
need not be silent, and overlapping speakers do not supply a clean cut;
neither a diarization label nor a low-energy frame guarantees a useful
recognition boundary.

Retain one packer and the same coverage, overlap, physical-length and
bounded-center-movement constraints. A diarizer proposes preferred
boundaries; it does not override full-slot occupancy or require a separate
encoding implementation. When no proposed boundary is feasible, retain
the ordinary schedule. A combined policy can use silence scores within
the feasible region around a predicted speaker change.

An offline diarizer may need the whole recording, substantial memory or
additional model storage. Measure those costs and its added latency,
including whether it can process long inputs without a new memory limit.
Keep it optional, with the cheap silence heuristic available; offline
diarization must not silently become a prerequisite for the
[hybrid streaming path](hybrid-chunked-streaming.md).

Compare uniform placement, silence placement, diarization placement and
their combination with the same encoder, decoder, window lengths and
overlap budget. Include both single-speaker talks and meetings with
speaker changes; report WER, end-to-end RTF including diarization, peak
memory and model storage. Inspect errors around proposed boundaries as
well as whole-recording WER. Select by recognition quality and total cost,
not diarization accuracy alone.

### Batch timing

Time the recognizer alone (the program prints it) on a grid of width by
length, each cell a list of one length decoded enough times to separate a
first call from repeats, per model and provider. Fit the smallest formula
that predicts a mixed-length run within the run-to-run spread; keep the
affine one if it does.

## Closure

A model fitted on such a grid predicts the recognizer time of a
mixed-length run it was not fitted on, and the program's cut uses it; or
the grid shows the affine formula with one measured overhead is within the
spread, and that overhead is the documented way to set the option.

Contributing-model: fable-5.2
Contributing-model: 6-Astra

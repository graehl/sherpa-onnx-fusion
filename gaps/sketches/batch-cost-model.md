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
  2026-11-15). Within a process nothing is: the cost recurs on every
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

## How to measure

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

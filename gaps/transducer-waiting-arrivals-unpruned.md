---
slug: transducer-waiting-arrivals-unpruned
noticed: 2026-11-18
where: transducer-prefix-search.cc (AddArrival, Prune, EndFrame, Outstanding, Finishable)
---

# TDT arrivals waiting for later frames are not pruned until they arrive

## Observation

The transducer prefix search prunes each queue to `max_active` within the
score beam when it is used: the frontier of a round (`again_`) and the
arrivals for a frame when that frame begins (`EndFrame` → `Prune`). An
arrival for a later frame (`AddArrival`, TDT durations 1 to
`max_duration`) joins its slot's list unpruned and pins its prefix until
then. Each round can add up to frontier × token candidates × durations
entries, and a frame has up to `max_symbols_per_frame` rounds, so the
waiting lists, and the prefixes and scorer states they pin, are bounded
only by those products.

On 40 LibriSpeech test-other recordings with Parakeet TDT 110M (int8, CPU)
and the test character LSTM (`max_active` 4, 8 candidates, beam 12), the
scorer had up to 11,822 LM nodes live, against 496 for the zipformer CTC
search with a larger LSTM, and its widest step took 6,045 nodes. My
current model of that step, unverified: `Finish` scores every outstanding
prefix (`Outstanding` includes every waiting arrival), so the end of an
utterance runs one step over all of them.

## Why pruning on insertion is not free

An entry's mass in its slot only grows until the slot's frame, as later
rounds and frames add paths into it, so an entry below the slot's best k
when inserted can still finish in them. Pruning on insertion by the same
rule is approximate; the question is how often it would drop an entry that
pruning on arrival keeps.

## Measured

`--trace-survival` (offline transducers) logs a table for the policy
"cut each waiting list at the end of every round to its best K within B
of its best". The same 40 recordings, 2026-11-18, with 35,730 round ends:

- **Now:** 2,337 entries wait at a round end on average, at most 3,179.
  They are mostly longer than their list's best: 36% by 8 tokens or more.
  Arrival pruning keeps entries within a token of the best (86% at -1 or
  0) and drops the long ones.
- **Every path of the best result survives** (a result charged for every
  path that added probability to it, however little): K 4 and B 12 keep
  45% of results unchanged; only K 512 with no gap limit keeps all 40.
- **The dominant path survives** (the path that at every queue brought
  the most probability): K 4 and B 12, the search's own `max_active` and
  `score-beam`, keep all 40, holding 14 entries per round end on average
  instead of 2,337. K 4 and B 8 keep 38 of 40.

Both are first-order: cutting other entries also changes what reaches
later queues. A dominant path that survives loses only the probability of
the paths it does not cover.

## What is wanted

Implement the cut as an option (K and B defaulting to `max_active` and
`score-beam`), and compare transcripts, error rate, live LM nodes and time
against the unbounded search on a corpus large enough to see a change.
`Finish` likewise need only score prefixes that can still rank, not every
waiting entry.

## Closure

Live prefixes and scorer states in a TDT search are bounded by a stated
policy whose effect on results is measured; transcripts unchanged or the
difference recorded.

Contributing-model: claude-opus-5-5

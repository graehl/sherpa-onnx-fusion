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

## What is wanted

Measure first: waiting entries per slot, and for each pruned-on-arrival
entry whether its insertion-time rank or gap to the slot's best so far
would have predicted that. Then choose a bound with a controlled
search-error rate, such as a beam against the slot's running best with a
margin, or a cap well above `max_active`, and compare transcripts and
live nodes against the unbounded search. `Finish` likewise need only score
prefixes that can still rank, not every waiting entry.

## Closure

Live prefixes and scorer states in a TDT search are bounded by a stated
policy whose effect on results is measured; transcripts unchanged or the
difference recorded.

Contributing-model: claude-opus-5-5

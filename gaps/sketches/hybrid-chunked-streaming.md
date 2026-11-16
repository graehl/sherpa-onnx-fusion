---
slug: hybrid-chunked-streaming
noticed: 2026-11-16
where: chunked encoding (offline-stream-encoder, offline-encoder-window), CtcInflightDecoder, transducer prefix search, recognizer orchestration
---

# Hybrid chunked streaming: search finished windows before the audio ends

## Intended outcome

User-directed (graehl, 2026-11-16): run an offline (whole-utterance)
encoder in windows as audio arrives, and let the search consume each
output frame as soon as it is final, instead of waiting for the whole
utterance to be encoded. This applies to models whose search needs only
the encoder frames up to the one it is on: CTC, and transducers, whose
search goes left to right over the frames. It does not apply to
attention encoder-decoder models, whose decoder attends to every encoder
frame.

Today [chunked encoding](../../topics/chunked-encoding.md) encodes every
window of an utterance and joins them before one search over the whole.
The search side already takes frames incrementally:
`CtcInflightDecoder::TryAccept` queues rows of log probabilities for an
utterance and `InputFinished` ends it.

## When a frame is final

An output frame is final when no later window will change it:

- with `--encoder.window.weight=center`, once every window that covers it
  has run, so a frame in the overlap waits for the next window;
- with `cut`, once the window it is taken from has run and the next
  window's start is fixed, since that start decides which window a frame
  in the overlap belongs to.

The search can run up to the last final frame, then wait.

## What blocks

- **Normalization over the utterance.** A feature normalization that uses
  statistics of its whole input (`--encoder.window.normalize=utterance`
  for w2v-BERT or NeMo `per_feature`) needs all the audio before the
  first window can run. Streaming needs per-window scope, or statistics
  of the audio so far (running or causal), which is a third choice to add
  to that option and to evaluate.
- **The schedule.** `EvenWindowStarts` needs the utterance's length: it
  spaces windows evenly, ends the last at the end, and looks at both
  edges of a window to place it in quiet audio. Streaming has to fix each
  window's start from the audio seen so far.

## A horizon between the two schedules

Make the schedule's lookahead an option, a continuum between:

- **go as soon as there is a full window:** each window starts at the
  previous start plus the step (window minus overlap), moved within the
  slack only as far as audio already received allows; the last window is
  placed when the input ends;
- **wait for all the audio:** today's schedule, decided once with the
  whole utterance.

For example `--encoder.window.horizon=SECONDS`: how much audio beyond a
window's end the scheduler may wait for before fixing that window, 0 for
the first, unbounded for the second. A longer horizon lets the scheduler
use the quiet-edge search on both edges, and, with the total length
known, space the windows evenly instead of leaving a short last window,
at the cost of latency. Both ends and the points between should be
measured for word error rate and for time to first and final result.

## Design notes

- The join already has the information: `WindowBlend` accumulates per
  frame, so finality is a property of which windows have been added. A
  streaming form needs an incremental `Finish` up to a frame.
- Batching changes shape: windows of different utterances become ready at
  different times, so encoder batches form from whatever windows are
  ready ([encoder-inclusive inflight scheduling](encoder-decoder-inflight-scheduling.md)).
- With `--encoder.lengths`, every full window runs at one shape as now.
  Only the last is short, and it is known when the input ends.
- Stateful streaming models (the fork's streaming zipformer, CTC and
  NeMo models) keep their own chunk-and-cache protocol; this is for
  models trained on whole utterances.

## Closure

A CTC model decoding a live or long input in windows returns its first
partial result after the first window instead of after the whole input,
with the final transcript identical to `horizon` unbounded whenever the
windows chosen are the same; and a measurement on long recordings of
error rate and latency across horizons.

Contributing-model: opus-5.5

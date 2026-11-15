---
slug: encoder-batches-ignore-length
noticed: 2026-11-14
where: offline recognizers' DecodeStreams and EncodeStreams, sherpa-onnx-offline, feature padding
---

# Encoder batches are formed without regard to length

## Observation

An offline decode call pads the features of every stream it is given to the
longest one and runs the encoder once. `sherpa-onnx-offline` passes every
file on its command line in one call. Nothing bounds the batch or groups
streams by length, so:

- **Memory grows with the number of recordings.** One call with a
  zipformer transducer peaked at 0.6 GB for 4 recordings, 3.4 GB for 16,
  and 9.4 GB for 48, about 200 MB each. A corpus run with 122 recordings
  per call exhausted a 124 GB host (2026-11-13).
- **Padding is computed and thrown away.** A batch of mixed lengths spends
  encoder time on the padded frames of every shorter recording. The waste
  was not measured.
- **Padding and batch composition change some transcripts**
  ([batch changes transcripts](offline-batch-changes-transcripts.md)), so
  an arbitrary batch is also a less reproducible one.

The fork's own batching is on the search side only: scorer calls across
utterances and hypotheses, and prediction-network and joiner calls across
hypotheses. None of it pads frames.

## What is wanted

User-directed (graehl, 2026-11-14): **batches of similar length under a
budget.** Sort the waiting streams by frame count, cut the sorted list into
chunks under a frame budget (longest member times batch size), run each
chunk, and return results in the caller's original order. The original
index travels with each stream through the sort. The batch is bounded by a
budget in frames or bytes, not by how many streams the caller passed.
`OfflineRecognizer::EncodeStreams` is the seam this sits behind, and it is
the encoder half of
[encoder-inclusive scheduling](sketches/encoder-decoder-inflight-scheduling.md).

A whole-utterance encoder gives no other handle. These encoders are not
recurrent over time: one call takes every frame of every utterance through
each layer in turn, and each frame of a layer depends on frames on both
sides of it in the layer below. No utterance finishes before the others,
so a running batch cannot shrink or admit a newcomer; its members are fixed
when it is formed. Membership that changes while work is in progress exists
in the search, and in chunked streaming encoders between chunks.

The aim is sensible encoder scheduling judged by average end-to-end
latency, each utterance's time from arrival to its result, and not by
encoder throughput alone. So the encoder is not run over everything before
any search starts: once a batch is encoded its utterances go to the search,
and encoding of the next batch and searching of the encoded ones are
interleaved (or overlapped where the device allows). Encoding first and
searching afterwards makes the first result wait for the last encode.

## Design notes

- The search side already keeps per-utterance state independent of batch
  position, and results are written back per stream, so reordering for the
  encoder does not disturb it.
- Models that cannot take a padded batch at all (a w2v-BERT CTC export
  with no length or mask input) stay at one utterance per call. The
  model class batches an export that takes `input_lengths`. Real masked
  w2v-BERT and omnilingual exports have matched single-item decoding in
  the checks recorded in [batch changes transcripts](offline-batch-changes-transcripts.md).
- Length bucketing makes padding small but not zero. Whether the remaining
  padding changes output is the other gap's question; a model whose export
  masks correctly makes the two independent.
- Sorting by length for small padding and serving in arrival order for low
  latency pull in different directions. A frame budget per batch with a
  bound on how long an utterance may wait for a better-matched batch is
  the usual compromise; the bound is a policy to measure, not to guess.
- Shortest-first ordering lowers average latency when everything has
  already arrived, and starves long utterances under steady load unless
  waiting time also counts.
- On a CPU the alternative to a batch is running items singly on separate
  threads (graehl, 2026-11-14). Optimized kernels can gain a little from a
  batch, but a large model run on one input already occupies the cores it
  is given, and when cores are free, one item per thread is often better.
  Which wins is only knowable once padding is out of the comparison: the
  one timing so far had batches of four carrying 34% more frames than the
  audio and running 29 to 35% slower than single items, which padding
  alone accounts for. Measure equal-length batches against threaded single
  items before choosing a batch size above one for CPU.
- The budget and the bucketing belong in the recognizer's `DecodeStreams`
  and `EncodeStreams`, so every caller gets them, including the language
  bindings; the command-line program should not need its own chunking.

## Closure

A decode call over many streams of mixed lengths uses memory bounded by the
budget and not by the stream count, returns results in the caller's order,
and spends measurably less encoder time than one padded batch; average and
worst time from arrival to result are reported for a mixed-length workload,
against encoding everything first; a test with
streams of very different lengths checks order and equality with decoding
each alone, for a model whose batched output is otherwise identical.

Contributing-model: fable-5.2

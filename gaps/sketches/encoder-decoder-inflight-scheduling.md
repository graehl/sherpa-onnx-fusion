---
slug: encoder-decoder-inflight-scheduling
noticed: 2026-11-13
where: recognizer orchestration, encoder batching, capture and executable drivers
---

# Encoder-inclusive inflight scheduling

## Intended outcome

User-directed: defer the overall encoder-inclusive use case while requiring
[the new decoder](../../topics/beam-search-and-scorer-fusion.md#inflight-batching)
to support inflight batching now. This sketch tracks future integration;
it does not claim that every current driver blocks capture while transcribing.

Allow capture of a following utterance while recognition of an earlier one
continues, including with offline ASR models. Capture/segmentation, encoder
execution, and decoder/scorer execution need independent queues, lifetimes,
and readiness decisions. Encoder and decoder/scorer batches need not contain
the same utterances or have the same sizes.

On a small device, encoder and decoder/scorer compute may alternate without
simultaneous network execution. That should not require disabling the mic
while showing “transcribing.” Pipeline progress across consecutive utterances
does not require overlapping GPU calls. Offline model input requirements
must be respected without treating them as a capture restriction.

## Seam already present

The offline CTC recognizer separates the two steps:
`OfflineRecognizer::EncodeStreams` runs the model and leaves each stream's
log probabilities on the stream, and `DecodeStreams` searches streams that
have them, running the model itself for any that do not. The two calls can
take different sets of streams, so a caller can encode an utterance when its
recording ends and search it later in a batch with others. Both calls still
block their caller, and nothing schedules them. Other model families ignore
`EncodeStreams`.

## Work deferred

- Define encoder-ready work and its handoff to decoder utterance handles,
  including chunked and complete-utterance models. The stream-held output
  above is the complete-utterance CTC case only; a transducer's search
  calls its prediction network and joiner, so its handoff carries encoder
  frames and the search still needs model time.
- Define queue limits, backpressure, fairness, cancellation, endpointing,
  and result delivery under sustained overload or interrupted input.
- Decide scheduling policy for serialized small-device compute and optional
  overlap, using latency, memory, and sustained thermal/power measurements.
- Integrate capture and UI behavior in the appropriate drivers, reusing
  infrastructure tracked in [the driver gap](../executable-driver-infrastructure.md).

## Completion evidence

Demonstrate consecutive utterances with uninterrupted capture while earlier
recognition remains active. Include serialized encoder/decoder execution,
unequal utterance durations, cancellation, and bounded-queue overload.
Verify results remain associated with the correct inputs and that retiring
one input does not invalidate another. Measure queue delay and resource use
before claiming a platform scheduling policy is suitable.

Contributing-model: gpt-6-astra

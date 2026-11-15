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

## Where encoder batch scheduling is headed

User-directed (graehl, 2026-11-15). This sketch is the plan's home; the
gaps hold measurements and defects.

The ambitions, beyond batching the encoder separately from the search:

- Keep encoder output on the device where the search can use it: no
  needless transfer to the host, and no copy out of the larger allocation
  a batch was encoded in.
- Search recently encoded utterances promptly, not after all encoding.
- Eventually a search that runs mostly on the GPU, which depending on the
  decoding type may be ambitious
  ([same-GPU sketch](../../topics/beam-search-and-scorer-fusion.sketches.md)).

The scheduling done so far answers one question only: how fast the
encoder result can be had when it is stored, printed or copied at once.
`sherpa-onnx-offline-parallel` sorts by length and cuts batches for
throughput, and [the batch cost model](batch-cost-model.md) refines that
with the cost of a change of shape
([measurements](../gpu-decode-cost-outside-the-model.md)).

Open: how the size of an encoder work unit interacts with those
ambitions. Known pulls, none measured:

- Serving at low latency cannot sort a whole corpus by length; it can
  only group what has arrived, within a waiting bound.
- A search that consumes encoder output on the device may prefer units
  sized for its own batch and memory, not the encoder's best shape.
- The shape-change cost favours long runs of one width and length, which
  is the opposite of taking utterances as they come.

## Completion evidence

Demonstrate consecutive utterances with uninterrupted capture while earlier
recognition remains active. Include serialized encoder/decoder execution,
unequal utterance durations, cancellation, and bounded-queue overload.
Verify results remain associated with the correct inputs and that retiring
one input does not invalidate another. Measure queue delay and resource use
before claiming a platform scheduling policy is suitable.

Contributing-model: gpt-6-astra

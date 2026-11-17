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
block their caller, and nothing schedules them. The zipformer and NeMo
transducers share the same step; other model families ignore
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

## Search batches apart from model batches

The API already lets them differ: `EncodeStreams` and `DecodeStreams` take
any sets of streams. `sherpa-onnx-offline-parallel` does not use that: with
or without `--decode-threads`, a search batch is exactly the model's batch
of `--batch-size`. They should eventually be chosen apart (graehl,
2026-11-17). The model's batch is set by padding and, on a GPU, by the cost
of a change of shape; the search's by how many utterances' scorer steps
share a call, where wider is cheaper per hypothesis and costs nothing in
padding. A model batch split into one or more search calls needs no
stitching, copy or care for lifetimes: each search call reads rows of an
output that outlives it. Only gathering several model batches into one
search call needs a copy or a reference count per allocation, so the
first step is a driver option for search calls of at most one model
batch. The search may also want its own pool of threads, as
`--decode-threads` already gives it; once any of its computation (a
scorer's step, a transducer's joiner) runs on a GPU while the search
itself stays on the CPU, which threads may touch which device's memory
becomes part of that choice (graehl, 2026-11-17).

## Handing the encoder output over by reference

What a stream keeps differs by family only in what the tensor holds: for
CTC the log probabilities, which are all the search needs; for the
zipformer and NeMo transducers the encoder frames, which the search feeds
to the joiner. Either way `SetEncoderOutput` copies the stream's rows out
of the batch's output into the stream. A stream could instead hold a
reference to the batch's whole output, its row and its valid length, the
copy made only when a later step needs contiguous rows. On the CPU the
search's own calls (an LSTM step, a transducer's joiner) have no
measurable shape-change cost, so this only saves the copy; on a GPU it is
what keeps the output where the search can use it, and the search's calls
would want the fixed widths of
[the fusion sketches](../../topics/beam-search-and-scorer-fusion.sketches.md#fixed-width-scorer-steps).

## Completion evidence

Demonstrate consecutive utterances with uninterrupted capture while earlier
recognition remains active. Include serialized encoder/decoder execution,
unequal utterance durations, cancellation, and bounded-queue overload.
Verify results remain associated with the correct inputs and that retiring
one input does not invalidate another. Measure queue delay and resource use
before claiming a platform scheduling policy is suitable.

Contributing-model: gpt-6-astra

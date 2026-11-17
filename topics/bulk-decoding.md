# Bulk decoding

> Decoding many recordings with `sherpa-onnx-offline-parallel`: length-sorted,
> bounded batches, results attributable to their recordings, model threads and
> search threads sized apart, and a check that none of this changes a
> transcript.

Topic: bulk-decoding

## What the program does

`sherpa-onnx-offline-parallel` reads a list of recordings (`--wav-scp`, or
the command line) and decodes them with `--nj` threads, each taking a
batch of up to `--batch-size` recordings at a time. It is the program for a
test set or a corpus; `sherpa-onnx-offline` pads every file of a call into
one batch.

- **Attributable results.** Each result is one JSON line on standard
  output beginning with `index`, the recording's position in the list, and
  `wav_filename`. A recording that cannot be read gets an `error` row.
  Rows are printed as they finish, or in list order with
  `--ordered-output=true`.
- **Batches of similar length.** By default recordings are decoded from
  the longest to the shortest (`--sort-by-length`), lengths estimated from
  the file size and wave header, so a batch needs little padding.
  `--max-batch-seconds` bounds a batch's padded audio: its longest
  recording times its count. For some models a recording's transcript
  depends on the others in its batch
  ([batching changes transcripts](../gaps/offline-batch-changes-transcripts.md)).
- **Model threads and search threads.** With `--decode-threads=N` the
  `--nj` threads only read files and run the model, and N other threads run
  the search, so a GPU model can take one thread while the search takes
  several; `--decode-queue-size` bounds the batches waiting between them.
  CTC models, the zipformer transducer and NeMo transducers separate the
  two steps (`OfflineRecognizer::EncodeStreams`); for other models the
  search threads run the model too. Scorers are shared across threads
  ([scorer sharing](beam-search-and-scorer-fusion.md)).
- **Timing.** It reports the recognizer's own time, excluding loading and
  `--warm-up` decodes; `--pause-seconds` leaves time to attach a monitor.

## Encoder output kept by reference

When the model runs ahead of the search, each stream shares its encoder
call's output (`EncoderOutputBatch`, `encoder-output.h`) and keeps its
item's index; nothing is copied into the stream, and the output is freed
when its last stream is searched. A search call whose streams are
consecutive items of one call, in order, reads that output in place
(`TakeEncoderOutputs`); any other mix is copied once into a padded tensor.
A call's padding, and windows of other utterances in the same call, stay
alive until its last stream is searched.

## The check

[`repro/check_parallel.py`](../repro/check_parallel.py) runs a CTC model, a
zipformer transducer and a Parakeet TDT over the recordings shipped with
the CTC model, with an unreadable file among them, in many search and
encoder-window configurations, and requires every way of running the
program (threads, search threads, sorting, printing as finished) to give
the same rows and texts as one thread. Some configurations must also have
an effect against a baseline. The fork's workflow runs it. Batches of
several recordings are checked for their rows only, since batching changes
what some models output; that sharing a search batch changes no transcript
is checked apart from the model, in `sherpa-onnx/python/tests/test_fusion.py`
([inflight batching](beam-search-and-scorer-fusion.md#inflight-batching)).
What the check does not cover is in [its gap](../gaps/parallel-check-coverage.md).

## Open work

- A timing against list-order batches on a GPU, and a measured per-call
  cost ([length-sorted batches](../gaps/parallel-driver-batches-in-list-order.md)).
- Length bucketing inside the recognizer's own `DecodeStreams`
  ([encoder batches ignore length](../gaps/encoder-batches-ignore-length.md)),
  and where GPU decode time goes
  ([GPU decode cost](../gaps/gpu-decode-cost-outside-the-model.md)).
- Shared executable-driver infrastructure
  ([driver reuse](../gaps/executable-driver-infrastructure.md)).
- A search batch chosen apart from the model's batch, and transducer
  searches that share their batch's model and scorer calls as CTC's do
  ([search batches](../gaps/sketches/encoder-decoder-inflight-scheduling.md#search-batches-apart-from-model-batches),
  [transducer search batching](../gaps/sketches/transducer-search-batching.md)).
- Candidate designs: [a measured batch cost model](../gaps/sketches/batch-cost-model.md),
  [encoder-inclusive inflight scheduling](../gaps/sketches/encoder-decoder-inflight-scheduling.md),
  and [the throughput table](../gaps/sketches/throughput-table.md).

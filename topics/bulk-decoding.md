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
  A w2v-BERT or omnilingual CTC model whose metadata has
  `batching_supported` (`true` or `false`, written by its exporter after
  decoding a padded batch against each recording alone) is batched or run
  one recording per call as that says; without it, such a model is
  batched only if it takes `input_lengths`. Padding is zero either way. An
  omnilingual model that says `true` but takes no `input_lengths` is
  rejected at load, since it cannot report each recording's valid frames.
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

## Throughput with an LSTM on one GPU

A sweep on 200 LibriSpeech test-other recordings (1,349 s), the zipformer
CTC model and icefall's LibriSpeech token LSTM (weight 0.3) fused by
prefix search, recognizer on CUDA, `--num-threads=2`, on an RTX PRO 6000
(96 GB) with 16 cores (2026-11-17). RTF is elapsed time from the end of
warm-up to the last result over audio time. Other work on the host held a
load of about 2 of 16 cores throughout, and nothing else used the GPU.
Repeated settings varied by at most 0.0002 (five runs of batch 32 with 4
threads, all 0.0015; four of batch 8 with one model thread and 4 search
threads, 0.0038–0.0040); single runs otherwise.

| LSTM on | Setting | RTF | Share of the best speed |
|---|---|---:|---:|
| GPU | `--batch-size=32 --nj=6` (best) | 0.0014 | 100% |
| GPU | `--batch-size=32 --nj=4` (fewest threads within 90%) | 0.0015 | 93% |
| GPU | `--batch-size=16 --nj=4` | 0.0017 | 82% |
| GPU | `--batch-size=32 --nj=2` | 0.0023 | 61% |
| GPU | `--batch-size=32 --nj=1` | 0.0038 | 37% |
| CPU | `--batch-size=16 --nj=2 --decode-threads=8` (best) | 0.0112 | 100% |
| CPU | `--batch-size=16 --nj=4` (fewest threads within 90%) | 0.0123 | 91% |

So placing the LSTM on the GPU makes the whole decode 8.0 times faster at
each placement's best setting, and 8.2 times at the cheapest within 90%.
On the GPU, threads that each run the model and search (`--nj`) beat one
model thread with separate search threads at every batch size (batch 32:
4 model threads 0.0015; 1 model thread with 1, 2, 4 or 8 search threads
0.0031–0.0035). Search calls sized apart from the model batch
(`--search-batch-size` 1 to 64) gave no speed in this setting: at each
model batch the best was within 0.0001–0.0002 of searching whole batches.
Batch 64 is no faster than 32 (0.0017 at 4 and 6 threads) and with 8
threads ran out of GPU memory; with the LSTM on the CPU that setting still
used 87 GB, so the recognizer's batched encoding is most of it
(`--max-batch-seconds` bounds it). Batch 32 with 4 threads peaked at 45 GB
of GPU memory with the LSTM there and 34 GB without.

## Search calls apart from model batches

`--search-batch-size=S` makes each search call take S recordings,
whatever the model's `--batch-size` (graehl, 2026-11-17). Each thread
that searches keeps the recordings whose model output is ready, in order,
and searches S at a time, the last call taking what remains. A model batch
is thus split into runs the search reads in place, and recordings left
over join the next batch's, at the cost of one copy of their output. The
search reads only each recording's valid frames, so S changes no result;
a size that divides `--batch-size` copies nothing. 0, the default, keeps a
model batch as one search call. Without `--decode-threads`, the model
thread runs the model for its whole batch and then searches in calls of S.
A model family that does not run its model apart from the search runs it
in the search calls instead.

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
an effect against a baseline. The fork's workflow runs it. Model batches
of several recordings are checked for their rows only, since batching
changes what some models output. That sharing a search call changes no
transcript is checked with the model at one recording per call and search
calls of three, against one thread, and in
`sherpa-onnx/python/tests/test_fusion.py`
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
- Transducer searches that share their batch's model and scorer calls as
  CTC's do ([transducer search batching](../gaps/sketches/transducer-search-batching.md)),
  and the placement of search threads once the search's own calls run on a
  GPU ([search batches](../gaps/sketches/encoder-decoder-inflight-scheduling.md#search-batches-apart-from-model-batches)).
- Candidate designs: [a measured batch cost model](../gaps/sketches/batch-cost-model.md),
  [encoder-inclusive inflight scheduling](../gaps/sketches/encoder-decoder-inflight-scheduling.md),
  and [the throughput table](../gaps/sketches/throughput-table.md).

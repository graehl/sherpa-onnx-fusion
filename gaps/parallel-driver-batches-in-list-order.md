---
slug: parallel-driver-batches-in-list-order
noticed: 2026-11-14
where: sherpa-onnx-offline-parallel (SplitToBatches, AsrInference)
---

# The parallel batch program batches in list order

## Observation

`sherpa-onnx-offline-parallel` reads a list of recordings (`--wav-scp`,
or the command line), cuts it into consecutive chunks of `--batch-size`,
and has `--nj` threads take chunks and decode each as one padded batch.
The chunks follow the list's order, so a batch holds whatever lengths
happen to be adjacent and every shorter recording is padded to the longest
in its chunk. Results are printed by each thread as its chunk finishes, so
their order is not the list's order when `--nj` is above one. Nor can they
be put back in order: a result is a JSON line on standard output with no
file name or index in it, the file name goes separately to standard error,
and nothing keeps one thread's pair of lines together (the program's mutex
guards only the timing totals). With several threads a transcript cannot
be reliably matched to its recording.

It is the one program here that bounds a batch, which would make it the
bulk driver if its output were usable: `sherpa-onnx-offline` decodes every
file of a call as a single batch
([encoder batches ignore length](encoder-batches-ignore-length.md)).

## What is wanted

User-directed (graehl, 2026-11-14): improve the driver, if needed, to
sort by length before batching. Read each recording's duration (the header
suffices), sort, cut into chunks of similar length, and write results in
the list's original order, carrying each recording's original index
through the sort.

"If needed" because the other gap puts length bucketing under a frame
budget inside the recognizer's `DecodeStreams`, where every caller and
binding gets it. Once that exists, this program needs only two things of
its own: pass the recognizer more streams than one batch so it has
something to bucket, and restore output order. Until then, sorting in the
driver is the cheap way to measure throughput without padding waste, which
the CPU batch-size question in that gap is waiting on.

## Design notes

- A chunk bounded by frames (longest member times count) fits better than
  a fixed count: a fixed `--batch-size` of long recordings is the memory
  problem, and of short ones underuses the batch.
- Sorting changes which recordings share a batch, so for models whose
  batched output depends on padding or on batch neighbours
  ([batch changes transcripts](offline-batch-changes-transcripts.md)) it
  changes some transcripts relative to list order. It makes padding small,
  which reduces the padding effect; it does not remove the int8
  neighbour effect.
- Each result record carries its list index and file name. Ordered output
  then needs results held until their turn, or collected and written in
  index order at the end; either way one writer at a time.
- Each thread's timer covers reading its chunk's files as well as
  decoding them, and the program sleeps ten seconds twice at startup.
  Separate file reading from decoding in what is reported when using it
  for a throughput comparison.

## State (2026-11-15)

Done in the program: each result is one JSON line naming its list index
and file, printed in list order for any `--nj`; `--sort-by-length` decodes
longest first; `--max-batch-seconds` bounds a batch by its longest file
times its count, beside `--batch-size`; the cut into batches is an exact
dynamic program for "a fixed overhead per batch plus the audio held after
padding" (`--batch-overhead-seconds`, by default so large that the fewest
batches are used), with ties going to equally full batches so no small
remainder is left; recognizer time is reported apart from file reading and
features. On 100 test-other recordings at 16 per batch, list order pads
697 s of audio to about 1480 s and sorting to 801 s.

Since then the program also runs the prefix search and scorers from
several threads, and `--decode-threads` gives it threads that only search,
fed through a bounded queue by the threads that run the model
([what a scorer holds](../topics/beam-search-and-scorer-fusion.md#scorer-interface)).

Still owed for closure:

- A timing on a GPU of list-order against sorted batches on a list of
  mixed lengths. Only CPU was run, under contention: summed recognizer
  time 46 s in list order and 27 s sorted.
- A measured overhead for a model and provider, to set
  `--batch-overhead-seconds` from something other than its default. The
  [GPU timing](gpu-decode-cost-outside-the-model.md) suggests tens to a
  couple of hundred seconds of audio per call for the zipformer CTC model
  on CUDA, depending on whether the input shape is new.
- The program still sleeps ten seconds twice at startup.

## Closure

Given a list of mixed lengths, the program's batches have padding below a
stated fraction of frames, results name their recording and come out in
list order for any `--nj`,
and a timing on such a list shows the encoder time saved against list-order
batches; or the recognizer does the bucketing and this program only
restores order, with the same evidence.

Contributing-model: fable-5.2

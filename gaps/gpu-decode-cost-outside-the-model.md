---
slug: gpu-decode-cost-outside-the-model
noticed: 2026-11-14
where: sherpa-onnx-offline-parallel, offline recognizer on the CUDA provider, feature extraction
---

# On a GPU most decode time is not model computation

## Observation

Timed 2026-11-14 with `sherpa-onnx-offline-parallel` built from the
upstream base (29b735c8) plus the output fix, CUDA provider, onnxruntime
1.28.2, an RTX PRO 6000 Blackwell, the public fp32 zipformer CTC model
`sherpa-onnx-zipformer-ctc-en-2023-10-02`, greedy search, LibriSpeech
test-other recordings of equal length so that batches carry almost no
padding. "Recognizer" is seconds inside `DecodeStreams`; "elapsed" also
covers reading each file and computing its features. Two runs each, both
shown where they differ; the GPU was otherwise idle.

48 recordings of 8 to 9 s (402 s of audio):

| Batch | Threads | Recognizer s | Elapsed s |
|---|---|---|---|
| 1 | 1 | 2.91–3.00 | 3.23–3.35 |
| 2 | 1 | 1.56 | 1.85 |
| 4 | 1 | 0.73 | 1.03 |
| 8 | 1 | 0.41–0.43 | 0.71–0.74 |
| 16 | 1 | 0.175 | 0.47–0.50 |
| 1 | 4 | 10.6–10.8 (summed) | 2.76–2.81 |
| 1 | 8 | 21.8–22.5 (summed) | 2.86–2.95 |
| 16 | 4 | 0.33–0.34 (summed) | 0.26–0.28 |

64 recordings of 3.25 to 3.75 s (224 s of audio) gave the same picture
more strongly: recognizer 4.06 s at batch 1 and 0.081 s at batch 16.

Three things stand out.

- **A call with a new input shape costs about 60 ms whatever is in it.**
  At batch 1 each call is about 60 ms for a 3.5 s recording and for an
  8.5 s one. Decoding one 8.5 s file 48 times, so the shape never changes,
  costs 13 ms a call (0.64 s in all). Roughly 47 ms of each
  varying-length call is therefore not computation. The cause is not
  identified. The cuDNN convolution search setting does not explain it:
  exhaustive and heuristic search timed alike and the fixed default was
  three times slower, in one run each while another job shared the GPU.
- **The cost is paid on every change of shape, not once per shape.**
  Decoding the 48-recording list one, two and four times over in one
  process at batch 1 took 5.1, 10.0 and 19.9 s of recognizer time
  (2026-11-15, another job on the GPU, so compare only among these). A
  second pass over shapes already seen is no cheaper than the first.
  So nothing is kept per shape inside the process, which rules out
  compilation per input size with a cache (graehl's first guess, by
  analogy with JAX) and leaves setup that is redone whenever the shape
  differs from the previous call's. Whether anything persists across
  processes was not tested and is unlikely to matter given this.
- **Batching itself is worth 4.2 times** at batch 16 for one shape (0.64 s
  against 0.153 s). The 17 to 50 times in the tables is mostly the
  per-shape cost being paid on fewer calls.
- **Feature extraction is the larger part once batches are large.** At
  batch 16 on one thread the recognizer is 0.175 s of 0.47 s elapsed; the
  rest is reading files and computing features on that one thread, about
  6 ms a recording. Four threads cut elapsed to 0.26 s by spreading that
  work, not by overlapping model calls.

Single items on several threads, all calling the one shared session, gain
little while shapes vary and something when they do not. With recordings
of different lengths, four threads at batch 1 spend 3.6 times the summed
recognizer time, so the model gets through its work about 1.1 times
faster than on one thread. Decoding one file 48 times, four threads spend
1.42 s summed against 0.64 s, about 1.8 times faster (one run). So the
per-shape cost appears to be serialized across threads, and with it
removed threaded single items still reach under half of what a batch of
16 does. Nothing in this repository takes a lock around the call; the
serialization is inside onnxruntime or the provider and was not traced.
The first call each thread makes is also slow on this
provider: warming up one thread of eight, not all eight, raised elapsed
from 1.81–1.93 s to 2.19–2.28 s on the short list (batch 2, three
alternating pairs).

On the CPU provider with one thread a batch of 8 was 12% slower than
single items (9.0 s against 8.0 s recognizer, one run), and eight threads
of single items finished in 1.55 s elapsed.

## What is wanted

- Find what the per-shape cost is and remove it or avoid it. Candidates:
  memory arena growth or reallocation per shape, per-shape kernel setup in
  the provider, shape-dependent work run on the CPU. Padding batch lengths
  up to a few fixed sizes would avoid it at the price of padded frames and
  whatever the padding does to transcripts
  ([batch changes transcripts](offline-batch-changes-transcripts.md)).
- Take feature extraction off the thread that calls the model, in a
  driver that means to keep a GPU busy: compute features on worker threads
  and hand finished streams to one batching caller.
- Repeat with a transducer and with the prefix search and a scorer, where
  the search is a larger share and the encoder gain a smaller one.

## Consequence for the other gaps

Similar-length batches do pay on this GPU, so the length sort in
[the parallel program gap](parallel-driver-batches-in-list-order.md) and
the bucketing in
[encoder batches ignore length](encoder-batches-ignore-length.md) are
worth building for it. On CPU the comparison that gap asked for came out
against batching.

## Closure

The per-shape cost is explained and a run over recordings of mixed
lengths spends recognizer time close to the same audio at one fixed shape;
a GPU driver's elapsed time is within a stated margin of its recognizer
time at the thread count it uses.

Contributing-model: fable-5.2

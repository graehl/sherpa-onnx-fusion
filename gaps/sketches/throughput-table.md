---
slug: throughput-table
noticed: 2026-11-15
where: repro/, sherpa-onnx-offline-parallel defaults, FORK.md
---

# A table of the best demonstrated throughput per kind of model

## Intended outcome

User-directed (graehl, 2026-11-15): a favorable performance table. For
each major kind of model, the configuration of `sherpa-onnx-offline-parallel`
with the largest demonstrated benefit (batch size, model threads, search
threads, sorting), beside a plain baseline. It has three jobs: show the
benefit, inform the program's defaults, and attest that the fast
configuration gives the right transcripts.

"Favorable" means each row shows the best configuration found, not that
unfavorable results are left out: a kind of model where batching or search
threads do not help gets a row saying so.

## Rows

One per kind of model and device, public models only in this repository
(private ones are measured in the research repository by the same method):

| Kind | Public model | Search |
|---|---|---|
| Zipformer CTC | `sherpa-onnx-zipformer-ctc-en-2023-10-02` | greedy; prefix search with an n-gram scorer; with an LSTM scorer |
| Zipformer transducer | `sherpa-onnx-zipformer-small-en-2023-06-26` or a larger one | greedy; upstream beam search; prefix search with an n-gram scorer |
| NeMo TDT or RNN-T conformer | a public Parakeet TDT export | greedy; prefix search |
| w2v-BERT CTC with a length input | none public yet | greedy; prefix search with scorers |

Each on CPU and on one GPU. An LSTM scorer strong enough to mean something
is needed for the fused rows; the pinned test LSTM is not.

## Columns

- Baseline: one thread, one recording per call, list order.
- Best configuration: `--batch-size`, `--nj`, `--decode-threads`,
  `--sort-by-length`, `--max-batch-seconds`, and where each scorer runs.
- Audio seconds per second for both, and their ratio; recognizer time
  apart from file reading and features.
- Correctness against the baseline: transcripts identical, or how many
  differ and the error rate of both. Batching changes some models' output
  ([gap](../offline-batch-changes-transcripts.md)), so "identical" is not
  assumed for a batch above one.
- Hardware, thread counts, the fork commit, and that the device was idle.

## Method

- One fixed list per language with mixed lengths, large enough that warm-up
  and the last partial batch do not dominate: LibriSpeech test-other, which
  `repro/` already pins.
- A sweep script in `repro/` that runs a grid, repeats each point three
  times, and writes one result file per row with every point, not only the
  winner. The table is generated from those files.
- An idle GPU. The timings taken so far on 2026-11-15 were not
  ([GPU gap](../gpu-decode-cost-outside-the-model.md)).
- Defaults follow from the table only where one configuration is best or
  near best across rows of a device; otherwise the program keeps its
  conservative defaults and the table is the guidance.

## What it waits on

`repro/check_parallel.py` covers CTC, transducer, and TDT configurations
([check coverage](../parallel-check-coverage.md)). Remaining prerequisites:

- Separate model and search calls for transducers, without which
  `--decode-threads` cannot help them
  ([sketch](separate-encode-for-other-model-families.md)).
- For the fused rows, a public LSTM of real quality.

## Closure

The table is in `FORK.md` or beside it, generated from committed result
files, every best configuration has a correctness entry, and the program's
defaults cite it.

Contributing-model: fable-5.2

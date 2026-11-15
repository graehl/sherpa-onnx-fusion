---
slug: public-lstm-fusion-evidence
noticed: 2026-11-14
where: topics/beam-search-and-scorer-fusion.md § Evidence; onnx-rnn scorer on transducer and CTC searches
---

# LSTM fusion lacks a result anyone can reproduce

## Observation

The [fusion evidence](../topics/beam-search-and-scorer-fusion.md#evidence)
includes plain WER for public English zipformer CTC and transducer models
with the public LibriSpeech LSTM over their tokens. Each comparison covers
600 recordings sampled from LibriSpeech test-other. The full standard
test sets, pinned model revisions, and public reproduction recipes required
below remain missing; `repro/claims.toml` has no LSTM claim entry.

## What is wanted

User-directed (graehl, 2026-11-14): one LSTM
fusion result for a transducer and one for a CTC model, each on models
anyone can download from Hugging Face, with a metric and test set anyone
can reproduce.

For each:

- **Models named by repository and revision**: the recognizer and the
  LSTM, with the exact files used and the option that maps the LSTM's units
  to the recognizer's (`--scorer.unit=token` with its `extra`, or character
  units with a unit list).
- **A standard test set, whole**: for English, LibriSpeech test-clean and
  test-other in full.
- **Plain WER** under a stated, public normalization (lowercase, no
  punctuation, as LibriSpeech references are written), computed by a tool
  the reader can run. ER2 may sit beside it; it cannot be the only number.
- **The same search without a scorer and the existing decoder** beside the
  fused row, and for the transducer the existing RNN-LM rescoring and
  shallow fusion, since those accept the same LM.
- **Settings chosen on the dev sets**, test read once; a paired test for
  the fused row against the unfused one.
- **Time**, one recording per call, with the batch width and thread count.
- **The commands**, in the topic or a script kept with it, that reproduce
  the table from the downloads.

## Design notes

- Both cases can extend the existing public token-LSTM comparisons to
  full standard sets. Pin the models and preserve their ID mapping with
  `--scorer.unit=token` and its `extra` options. Check token tables when
  substituting a model; a vocabulary of the same size is not the same vocabulary
  ([vocabulary compatibility](sketches/scorer-vocabulary-compatibility.md)).
- Decode one recording per call so that batching does not enter the
  comparison ([batch changes transcripts](offline-batch-changes-transcripts.md)),
  and run corpus-sized jobs outside any interactive session with a memory
  cap.
- An n-gram row on the same models costs little and shows the LSTM's gain
  over the cheaper scorer.

## Closure

The topic's evidence section has, for one public transducer and one public
CTC recognizer, a table with the rows above on full standard test sets in
plain WER, with model revisions and commands, and a reader with the
downloads can reproduce it.

Contributing-model: fable-5.2

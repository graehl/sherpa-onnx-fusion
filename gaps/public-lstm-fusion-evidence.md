---
slug: public-lstm-fusion-evidence
noticed: 2026-11-14
where: topics/beam-search-and-scorer-fusion.md § Evidence; onnx-rnn scorer on transducer and CTC searches
---

# LSTM fusion lacks a result anyone can reproduce

## Observation

The fusion topic's LSTM results cannot be checked from outside:

- **Transducer.** One run uses only public files: the English zipformer
  transducer (`sherpa-onnx-zipformer-en-2023-06-26`) with the public
  LibriSpeech RNN LM over its 500 tokens. It covers 600 recordings sampled
  from LibriSpeech test-other, and its scores are ER2, from a scorer that
  is not in this repository. The sample is not a standard set and the
  metric is not one a reader can compute.
- **CTC.** No LSTM result on a public CTC recognizer exists at all. The
  CTC LSTM runs used recognizers and language models that are not public.

## What is wanted

User-directed (graehl, 2026-11-14; recorded, not yet to be run): one LSTM
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

- The transducer case is mostly a rerun of what exists, on the full sets
  and with WER.
- The CTC case needs an LSTM whose units a public CTC recognizer can use.
  Two routes, neither verified: a public CTC model trained with the same
  500-token vocabulary as the LibriSpeech RNN LM, used with
  `--scorer.unit=token`; or a character LSTM trained here from public text
  and published, which works with any CTC token table. Check the token
  tables for equality before assuming the first; a vocabulary of the same
  size is not the same vocabulary
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

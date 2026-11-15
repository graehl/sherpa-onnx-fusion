---
slug: parallel-check-coverage
noticed: 2026-11-15
where: repro/check_parallel.py, .github/workflows/fusion-fork.yaml, .github/workflows/linux.yaml
---

# What the parallel program check does not cover

## Observation

`repro/check_parallel.py` compares the ways of running
`sherpa-onnx-offline-parallel` with one thread. It is the only repeatable
check of running the search and scorers from several threads. It leaves
out:

- **Model families other than CTC.** A transducer was checked by hand
  (2026-11-15: the small public English zipformer, 61 recordings; greedy,
  upstream's beam search, the prefix search, and the prefix search with
  KenLM; 8 threads, sorted order, and search threads all matched one
  thread). Nothing repeats that.
- **An LSTM scorer by default.** It runs only with `--lstm-model`. The
  one public recurrent LM in ONNX form that upstream's tests use is a
  whole-sentence export (about 400 MB, 103 MB quantized) without the state
  inputs the `onnx-rnn` scorer steps through, and no small step-wise one
  was found to pin.
- **A wrong transcript.** The check was seen to fail on a build that
  lacks its options, never on output that differs.
- **The build without KenLM.** That it skips the n-gram configuration
  was checked only by the message such a build prints.
- **A GPU.** Both workflows are CPU only.
- **Either workflow actually running.** Neither has.

## What is wanted

- Pin a small step-wise LSTM and run it by default. A tiny English
  character one (one layer of 256, about 2 MB) is being trained for this
  (2026-11-15). It is to be hosted on Hugging Face and pinned by URL and
  hash; no model is committed to this repository.
- Add a transducer configuration; the small English zipformer archive is
  112 MB.
- A deliberate fault (for example an environment variable in a test build
  that swaps two results) to see the check fail on wrong output.

## Closure

The default run covers a transducer and an LSTM scorer, and the check has
been seen to fail on a wrong transcript.

Contributing-model: fable-5.2

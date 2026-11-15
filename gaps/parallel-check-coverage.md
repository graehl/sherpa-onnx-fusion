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
- **A real LSTM.** The default run uses a 1.6 MB character LSTM trained
  for 15 minutes as a test fixture
  ([graehl/sherpa-onnx-fusion-test-lm](https://huggingface.co/graehl/sherpa-onnx-fusion-test-lm)).
  It exercises the scorer; it says nothing about recognition with a
  language model. A model over the recognizer's tokens runs only with
  `--lstm-model`.
- **A wrong transcript.** The check was seen to fail on a build that
  lacks its options, never on output that differs.
- **The build without KenLM.** That it skips the n-gram configuration
  was checked only by the message such a build prints.
- **A GPU.** Both workflows are CPU only.
- **Either workflow actually running.** Neither has.

## What is wanted

- Add a transducer configuration; the small English zipformer archive is
  112 MB.
- A deliberate fault (for example an environment variable in a test build
  that swaps two results) to see the check fail on wrong output.

## Closure

The default run covers a transducer, and the check has been seen to fail
on a wrong transcript.

Contributing-model: fable-5.2

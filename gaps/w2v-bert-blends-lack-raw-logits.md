---
slug: w2v-bert-blends-lack-raw-logits
noticed: 2026-11-24
where: offline-encoder-window.cc (WindowBlend), w2v-BERT ONNX and LiteRT exports (k2-fsa scripts/w2v-bert/export_onnx.py; ~/draft scripts/asr_ctc_onnx_export.py)
---

# Window blends that need raw logits cannot be tried on w2v-BERT exports

## Observation

The chunked-encoding window blend was meant to allow experiments with
different ways of joining overlapping windows' CTC outputs (graehl,
2026-11-24). Since 2026-11-24, w2v-BERT exports from both exporters end in
a log-softmax: k2-fsa's export, at its maintainer's request in
k2-fsa/sherpa-onnx#4263, names the outputs `log_probs` and
`log_probs_lengths` (metadata version 2), and the research exporter
followed. A graph no longer returns the logits themselves.

What is lost is one number per window and frame: the log-sum-exp \(c\)
of the frame's logits, since log probabilities are the logits less \(c\).
Every blend built today is unaffected, exactly: the center-weighted linear
blend renormalizes after averaging, which removes any per-window constant
(\(\operatorname{lsm}(\sum_i w_i (z_i - c_i) / \sum_i w_i) =
\operatorname{lsm}(\sum_i w_i z_i / \sum_i w_i)\); numerically within
1.8e-15 on random windows), `blend=prob` normalizes each row itself, and
`cut` copies one row. So are blends added later that ignore a per-window
constant: temperature, entropy or margin weighting.

A blend that uses the logits' scale cannot be tried: weighting a window by
its energy (\(c\) itself, an out-of-distribution signal), by its maximum
raw logit, or averaging unrenormalized logits.

## What is wanted

When such a blend is to be tried, obtain raw logits for the experiment
without changing the default exports:

- **ONNX:** rewrite a copy of the model file so the LogSoftmax node's
  input is also a graph output (or the node is removed). The runtime
  checks for exactly two outputs, so a third needs the w2v-BERT model to
  ignore or read it.
- **LiteRT:** add an exporter option (say `--raw-logits`) to
  `scripts/asr_ctc_onnx_export.py` that adds a `logits` output, and let
  `offline-w2v-bert-ctc-litert-model.cc` accept the third output. A
  compiled TPU graph cannot be edited and must be compiled again from the
  new portable graph. Every LiteRT output is computed and copied out on
  every call: \(4FV\) bytes more per call (about 100 KB at 30 s for the
  Galician model's few dozen characters), more for a large vocabulary.

Do not export both by default until a blend needs them.

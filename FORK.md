# Differences from k2-fsa/sherpa-onnx

This fork aims for efficient phone and server ASR inference with LM fusion,
adding CTC support and improving use cases that already have fusion support.
It develops shared search components with pluggable scorers for CTC,
transducer, and autoregressive models, starting with sum-over-alignments CTC
prefix search. The decoder is currently in design; its proposed
interfaces and CLI options are not implemented.

Start with the [shared beam search and scorer fusion topic](topics/beam-search-and-scorer-fusion.md).
It records the evidence, the first implementation's scope, the C++ interface
sketch, and configuration; designs for later passes are in its
[sketches companion](topics/beam-search-and-scorer-fusion.sketches.md), and
the fork's own terms in the [glossary](GLOSSARY.md). The design follows and
extends the
[LM-fusion proposal in upstream issue #4186](https://github.com/k2-fsa/sherpa-onnx/issues/4186).

## Intended decoder additions

- Sum CTC alignment probabilities by output prefix, with or without an LM.
- Fuse multiple independently weighted scorers during search, supporting
  token or character units and factories with backend-specific configuration.
- Follow the existing C++ interface conventions, with batched scorer calls
  and utterance-owned storage independent of batch membership.

## Current code difference in this work

The [streaming CTC FST characterization test](sherpa-onnx/csrc/online-ctc-fst-decoder-test.cc)
reproduces trailing-silence overcount across chunks and its effect on
endpointing. It deliberately passes when the current incorrect behavior is
reproduced and prints a warning describing the correct result. The decoder
defect is not fixed by that test.

## Tracked design and maintenance work

- [Streaming CTC trailing-silence overcount](gaps/streaming-ctc-trailing-blank-overcount.md),
  the defect the characterization test records.
- [Repeated streaming CTC traceback work](gaps/streaming-ctc-traceback-recomputation.md).
- [Shared beam options and tuning guidance](gaps/beam-search-options-and-tuning.md).
- [Executable-driver infrastructure reuse](gaps/executable-driver-infrastructure.md).
- [Typed collections of nested options](gaps/sketches/nested-option-collections.md),
  including multiple scorers and possible multiple-FST configuration.
- [Encoder-inclusive inflight scheduling](gaps/sketches/encoder-decoder-inflight-scheduling.md),
  deferred beyond the decoder's own inflight batching.

## Relationship to upstream

Keep changes focused and close to `k2-fsa/sherpa-onnx` conventions so future
upstream merges stay low friction. Implement and prove the decoder here
first. Contributions to Sherpa or Kaldi remain possibilities rather than
prerequisites or commitments. The main README retains upstream's general
feature, platform, and usage information below the fork notice.

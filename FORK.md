# Differences from k2-fsa/sherpa-onnx

This fork aims for efficient phone and server ASR inference with LM fusion.
It adds a prefix beam search with pluggable scorers, shared across model
families, and follows and extends the
[LM-fusion proposal in upstream issue #4186](https://github.com/k2-fsa/sherpa-onnx/issues/4186).

Start with the [shared beam search and scorer fusion topic](topics/beam-search-and-scorer-fusion.md):
it says what is built, names each part's source file, and records the
evidence. Designs for later work are in its
[sketches companion](topics/beam-search-and-scorer-fusion.sketches.md), and
the fork's own terms in the [glossary](GLOSSARY.md).

## What the fork adds

- `--decoding-method=prefix_beam_search` for offline and streaming CTC
  models, offline and streaming stateless-decoder transducers (zipformer),
  and offline NeMo RNN-T and TDT models. CTC can sum over alignments or take
  the best.
- Settings that reproduce the existing decoders exactly:
  `--modified-beam-search-rules` for `modified_beam_search`, and a scorer's
  `rescore` option for RNN-LM rescoring. The topic's "Goals reached" table
  has the measurements.
- Scorers fused during the search, several at once with independent
  weights: a KenLM character n-gram (`-DSHERPA_ONNX_ENABLE_KENLM=ON`, off by
  default because KenLM is LGPL) and an ONNX LSTM or GRU over characters or
  over the recognizer's tokens.
- Scorer configuration on the command line (`--scorer.type=kenlm
  --scorer.model=lm.bin`, or `--scorer.N.<option>` for several) and from a
  JSON file ([option collections](topics/json-option-collections.md)).
- A multi-utterance CTC decoder that admits, feeds, finishes, and cancels
  utterances individually while batching scorer calls across them.

## Known defect characterized, not fixed

The [streaming CTC FST characterization test](sherpa-onnx/csrc/online-ctc-fst-decoder-test.cc)
reproduces trailing-silence overcount across chunks in the existing FST
decoder and its effect on endpointing. It passes while the defect is
present and prints a warning describing the correct result.

## Tracked design and maintenance work

- [Streaming CTC trailing-silence overcount](gaps/streaming-ctc-trailing-blank-overcount.md),
  the defect the characterization test records.
- [Repeated streaming CTC traceback work](gaps/streaming-ctc-traceback-recomputation.md).
- [Shared beam options and tuning guidance](gaps/beam-search-options-and-tuning.md).
- [Executable-driver infrastructure reuse](gaps/executable-driver-infrastructure.md).
- [Typed collections of nested options](gaps/sketches/nested-option-collections.md),
  including multiple scorers and possible multiple-FST configuration.
- [Language bindings for the fusion options](gaps/fusion-config-bindings.md).
- [Static build aborts when a model loads (GCC 15)](gaps/static-onnxruntime-abort.md).
- [Decoding several recordings in one call changes some transcripts](gaps/offline-batch-changes-transcripts.md).
- [Encoder batches are formed without regard to length](gaps/encoder-batches-ignore-length.md).
- [The parallel batch program batches in list order](gaps/parallel-driver-batches-in-list-order.md).
- [Encoder-inclusive inflight scheduling](gaps/sketches/encoder-decoder-inflight-scheduling.md),
  deferred beyond the decoder's own inflight batching.
- [Tuning fusion weights and bonuses from the search's own evidence](gaps/sketches/fusion-parameter-tuning.md).
- [Prefix search against the FST decoder on a graph with an LM](gaps/sketches/fst-decoder-lm-graph-comparison.md).
- [Checking that a token scorer and a recognizer share a vocabulary](gaps/sketches/scorer-vocabulary-compatibility.md).
- [Running models through LiteRT](gaps/sketches/litert-model-backend.md).

## Relationship to upstream

Keep changes focused and close to `k2-fsa/sherpa-onnx` conventions so future
upstream merges stay low friction. Implement and prove the decoder here
first. Contributions to Sherpa or Kaldi remain possibilities rather than
prerequisites or commitments. The main README retains upstream's general
feature, platform, and usage information below the fork notice.

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

## WER on public models

Word error rate in percent. Every recognizer here can be downloaded ([recipes](repro/)).

| Recognizer | Test set | Language model | LM MB | Greedy | Best k2-fsa sherpa-onnx decoder, with LM | Prefix search, no LM | Prefix search, with LM | RTF greedy | RTF prefix search, no LM | RTF prefix search, with LM | RTF k2-fsa decoder, no LM | RTF k2-fsa decoder, with LM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vietnamese Parakeet CTC 0.6B (NVIDIA, int8) * | FLEURS Vietnamese (857) | character 8-gram | 144 | 9.27 | § | 9.09 | 6.30 | 0.030 | 0.028 | 0.030 | 0.030 | – |
| English zipformer CTC | LibriSpeech test-other (2,939) | character 8-gram (LibriSpeech LM text) | 16 | 5.65 | 5.39 ◊ | 5.65 | 5.11 | 0.015 | 0.015 | 0.016 | 0.015 | 0.019 |
| the same * | LibriSpeech test-other (600) | LibriSpeech LSTM, ASR-tokens (icefall's `rnn_lm`) | 407 | 5.63 | ¶ | 5.59 | 4.45 | 0.021 | 0.015 | 0.13 | 0.021 | – |
| Vietnamese zipformer transducer (gipformer 1.5, 68M) * | FLEURS Vietnamese (857) | 4-gram, ASR-tokens | 108 | 5.59 | ‖ | 5.49 | 5.23 | – | 0.021 | 0.021 | – | – |
| the same * | FLEURS Vietnamese (857) | LSTM, ASR-tokens (int8) | 51 | 5.59 | 5.21 (rescoring, with the fp32 LSTM) | 5.49 | 4.89 | – | 0.021 | 0.028 | – | 0.049 |
| English zipformer transducer * | LibriSpeech test-other (600) | LibriSpeech LSTM, ASR-tokens (icefall's `rnn_lm`) | 407 | 4.65 | 4.38 (rescoring) | 4.60 | 4.20 | 0.059 ‡ | 0.063 ‡ | 0.51 ‡ | – | 0.13 ‡ |
| GigaAM v3 RNN-T * | FLEURS Russian (743) | character 8-gram from Russian Wikipedia | 907 | 4.63 | # | 4.52 | 4.21 | 0.067 | 0.25 † | 0.22 | – | – |
| Parakeet TDT 0.6B v3 * | FLEURS Russian (743) | character 8-gram from Russian Wikipedia | 907 | 8.32 | # | 7.77 | 6.78 | 0.25 | 0.69 † | 0.22 | – | – |

RTF is the recognizer's own decode seconds over audio seconds, summed over
calls of one recording each, two threads per call, model loading
excluded. The English CTC row was timed on an x86 host with 8 physical
cores (Xeon Platinum 8559C, 16 hardware threads), two calls at a time,
with at most 8 hardware threads busy, counting other users, in every
minute of the runs its figures come from. \* Timed while several
two-thread processes shared the host with other work, so rough; to be
rerun the same way, with peak memory, which is not measured yet. † Taken before the search batched its prediction-network and
joiner calls; on four recordings afterwards it ran at 2.4 and 1.8 times
greedy's time. ‡ This row was timed under about three times the load of
the others (compare its greedy figure with the CTC row's); the same LSTM
fused on the CTC model reads 0.13 under the rough conditions.

k2-fsa sherpa-onnx's decoder without an LM is greedy search for a CTC
model, so its RTF is greedy's, and `modified_beam_search` for a
transducer, not timed here; – means not measured. Where k2-fsa
sherpa-onnx has no decoder for the row's LM: § it takes an n-gram only as a
compiled graph, and not for NeMo CTC models; ¶ it has no RNN LM for CTC
models; ‖ it has no n-gram LM for transducers; # it has no LM for NeMo
models. ◊ With k2-fsa's word 3-gram graph released with the model
(350 MB), not the row's LM. LM MB is the size of the file the scorer
loads, in millions of bytes.

- **Language models.** The two Vietnamese models are not public. Both were
  built from 29.5 million sentences: Vietnamese Wikipedia, FineWeb-2, and
  transcripts or captions of GigaSpeech2, VietSpeech, VLSP 2020 and YODAS2
  with some private training transcripts, after removing sentences that
  match the test references. The gipformer LSTM was trained on the same
  sentences. The Russian 8-gram is from 55 million Wikipedia sentences with
  FLEURS validation and test references removed (907 MB, not yet cut down
  to the recognizers' alphabet); with only 49 thousand Common Voice and
  FLEURS training sentences the same two recognizers read 4.45 and 7.72.
- **Token or character.** On the English CTC model an n-gram over the
  recognizer's own tokens and a character n-gram from the same text give
  the same WER (4.91 each; 126 MB and 37 MB), and the LSTM is better than
  either.
- **A compiled graph against a scorer.** For a CTC model the way k2-fsa
  sherpa-onnx takes an n-gram is a decoding graph built ahead of time. The same character
  3-gram is 30 KB as a KenLM file and 417 MB compiled, with the same WER
  either way; a 6-gram's graph did not build in 24 GB. Both English CTC
  results above differ from that released graph's at p = 0.002 or better.
- **Nothing lost by switching.** The prefix search reproduces
  `modified_beam_search`, RNN-LM rescoring and streaming shallow fusion
  transcript for transcript, and its streaming RNN-LM fusion gave the same
  40 transcripts in 59 s against 309 s.
- **CTC without an LM.** On Citrinet-512 it finds the same best alignment
  as the graph decoder in 5 to 23 µs per frame against 380 to 3200, the
  graph holding no LM.

[Shared beam search and scorer fusion](topics/beam-search-and-scorer-fusion.md)
gives settings and paired tests (§ Fusion on a public CTC model, § Fusion
on a transducer, § Scorers over the recognizer's tokens, § Fusion on RNN-T
and TDT, § Matching the existing decoders).

## In this fork

### Search and LM fusion

- `--decoding-method=prefix_beam_search` for offline and streaming CTC
  models, offline and streaming stateless-decoder transducers (zipformer),
  and offline NeMo RNN-T and TDT models. CTC can sum over alignments or take
  the best.
- Scorers fused during the search, several at once with independent
  weights: a KenLM character n-gram (`-DSHERPA_ONNX_ENABLE_KENLM=ON`, off by
  default because KenLM is LGPL) and an ONNX LSTM or GRU over characters or
  ASR-tokens, configured on the command line
  (`--scorer.type=kenlm --scorer.model=lm.bin`, or `--scorer.N.<option>` for
  several) or from a JSON file
  ([option collections](topics/json-option-collections.md)).
- An LSTM on a GPU (`--scorer.N.provider=cuda`) keeps its recurrent states
  there. With icefall's LibriSpeech LSTM fused into the English CTC model,
  200 test-other recordings decoded at RTF 0.0014 with the LSTM on the GPU
  against 0.0112 with it on the CPU, 8 times faster, at the best of a sweep
  of batch sizes and threads on one RTX PRO 6000
  ([bulk decoding](topics/bulk-decoding.md#throughput-with-an-lstm-on-one-gpu)).
- Settings that reproduce the existing decoders exactly:
  `--modified-beam-search-rules` for `modified_beam_search`, and a scorer's
  `rescore` option for RNN-LM rescoring. The topic's "Goals reached" table
  has the measurements.
- A multi-utterance CTC decoder that admits, feeds, finishes, and cancels
  utterances individually while batching scorer calls across them.

### Models and encoding

- [Chunked encoding](topics/chunked-encoding.md), off by default: an
  utterance longer than `--encoder.window.seconds` is encoded in
  overlapping windows, batched, and joined per output frame before one
  search, for the offline CTC models, the zipformer transducer and NeMo
  RNN-T and TDT; `--encoder.lengths` and `--encoder.pad-to` keep encoder
  inputs to a few lengths for a GPU or NPU. On a 92 s recording, 20 s
  windows cut peak memory 1.6 to 3.7 times on three public models.
- w2v-BERT CTC models in LiteRT's `.tflite` form
  (`-DSHERPA_ONNX_ENABLE_LITERT=ON`, off by default), including a graph
  compiled for a Pixel 10 or 11, run on its TPU with `--provider=npu`
  ([topic](topics/litert-backend.md)). A graph has fixed
  lengths, one per signature, and several graphs may be given: an
  utterance runs in the shortest length that holds it, a longer one in
  blended overlapping windows (`--litert-window`).
- Batched decoding for w2v-BERT and omnilingual CTC exports that take
  `input_lengths`, whose fp32 results then do not depend on the batch.

### Bulk decoding

[`sherpa-onnx-offline-parallel`](topics/bulk-decoding.md) decodes a list
of recordings; `sherpa-onnx-offline` itself still pads every file on its
command line into one batch.

- One JSON line per recording naming its position in the list, with an
  error line for a file that cannot be read, printed as each finishes or
  in list order (`--ordered-output`).
- Batches of similar length by default (`--sort-by-length`) under a
  padded-audio limit (`--max-batch-seconds`).
- For CTC models and the zipformer and NeMo transducers, search threads
  sized apart from the threads that run the model (`--decode-threads`),
  and search calls sized apart from model batches (`--search-batch-size`).
  Scorers are shared across threads.

### Language bindings

Every [language binding](topics/language-bindings.md) exposes these
options: the C and C++ APIs, Python, Java and Kotlin, Go, Rust, C#,
Pascal, Swift, Node.js, HarmonyOS, WebAssembly and Dart. Scorers and
search settings are typed fields of a `fusion` config on both recognizer
configs, the encoder and LiteRT settings and the w2v-BERT model are fields
of the offline model config, and offline recognizers have encode methods
beside their decode methods. Any command-line option can also be given as
`options_json`, a JSON object applied after the typed fields. The
multi-utterance decoder and the bulk program's scheduling have no binding.

### Reproducing the results

[repro/](repro/README.md) holds pinned public downloads and a script that
reruns the English CTC comparisons above, and a check that threads, search
threads, search calls and sorting do not change the bulk program's
output.

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
- [What the language bindings' checks did not cover](gaps/fork-feature-bindings.md).
- [A public inflight decoder API](gaps/public-inflight-decoder-api.md)
  for every decoder that can keep utterances in flight.
- [Static build aborts when a model loads (GCC 15)](gaps/static-onnxruntime-abort.md).
- [Inherited build parts that outside changes broke](gaps/upstream-ci-toolchain-drift.md):
  Flutter's Gradle minimum and the vanished Microsoft ONNX Runtime downloads.
- [Decoding several recordings in one call changes some transcripts](gaps/offline-batch-changes-transcripts.md).
- [Encoder batches are formed without regard to length](gaps/encoder-batches-ignore-length.md).
- [Length-sorted batches in the parallel batch program](gaps/parallel-driver-batches-in-list-order.md):
  built; a GPU timing against list-order batches and a measured per-call
  overhead are still owed.
- [What the parallel program check does not cover](gaps/parallel-check-coverage.md).
- [On a GPU most decode time is not model computation](gaps/gpu-decode-cost-outside-the-model.md).
- [Separate model run and search for families other than CTC](gaps/sketches/separate-encode-for-other-model-families.md).
- [A measured cost model for a batch](gaps/sketches/batch-cost-model.md).
- [A table of the best demonstrated throughput per kind of model](gaps/sketches/throughput-table.md)
  (no rows yet: no timing has been taken on an idle machine).
- [LSTM fusion lacks a result anyone can reproduce](gaps/public-lstm-fusion-evidence.md).
- [Encoder-inclusive inflight scheduling](gaps/sketches/encoder-decoder-inflight-scheduling.md),
  deferred beyond the decoder's own inflight batching.
- [Tuning fusion weights and bonuses from the search's own evidence](gaps/sketches/fusion-parameter-tuning.md).
- [Prefix search against the FST decoder on a graph with an LM](gaps/sketches/fst-decoder-lm-graph-comparison.md) (run for a character LM; what remains is in the sketch).
- [Checking that a token scorer and a recognizer share a vocabulary](gaps/sketches/scorer-vocabulary-compatibility.md).
- [Running models through LiteRT](gaps/sketches/litert-model-backend.md).
- [Hybrid chunked streaming](gaps/sketches/hybrid-chunked-streaming.md):
  searching an offline model's finished windows before the audio ends.
- [A remaining-cost estimate for partly spelled words](gaps/sketches/word-lm-remaining-cost.md).

## Relationship to upstream

Keep changes focused and close to `k2-fsa/sherpa-onnx` conventions so future
upstream merges stay low friction. Implement and prove the decoder here
first. Contributions to Sherpa or Kaldi remain possibilities rather than
prerequisites or commitments. The main README retains upstream's general
feature, platform, and usage information below the fork notice.

---
slug: asr-token-output-api
noticed: 2026-11-19
where: OfflineRecognitionResult, the recognizer impls' Convert, c-api, Rust and other bindings
---

# Recognizer token IDs and token table for callers outside sherpa

## Intended outcome

A caller that processes the recognizer's output in its own code can work
on the recognizer's tokens as integers, against the token table sherpa
already loaded, with no second copy of the vocabulary and no string
round trip (graehl, 2026-11-19). Two things are wanted:

- **Output token IDs.** Each result carries `token_ids`, parallel to
  `tokens`, `timestamps`, `durations` and `ys_log_probs`. They are the
  IDs the search emitted, taken before anything sherpa does to the text
  afterwards (rule FSTs and FARs, the homophone replacer, punctuation),
  none of which changes the token sequence.
- **The loaded token table.** The recognizer exposes the table it
  decodes with: size, the symbol for each ID (raw piece bytes, as after
  Whisper's base64 decoding), and which IDs are blank, unknown and the
  sentence boundaries. Callers get it once per recognizer, not with each
  result.

Today a result has only the token strings, and a caller who wants IDs
re-reads `tokens.txt` and inverts it. That holds a second copy and
assumes the symbols are unique, which is unchecked for byte-level
vocabularies such as Whisper's.

## Design notes

- **Where the IDs are lost.** The decoders' results hold IDs; each
  recognizer type's `Convert` in `sherpa-onnx/csrc/offline-recognizer-*impl.h`
  (14 files) maps them to strings. Each gains one line copying the IDs.
  The online recognizers need the same. A model family whose `tokens`
  are not one per emitted ID (merged pieces, dropped language or task
  tokens) must say so or leave `token_ids` empty.
- **C API.** `token_ids` goes in the result JSON, and as an `int32_t`
  array with its count on the result struct, so that a caller on the hot
  path does not parse JSON. The table is a new pair of functions on the
  recognizer: a count, and the symbol for an ID, with a borrowed pointer
  valid while the recognizer lives. A digest of the table, as in
  [scorer-vocabulary-compatibility](scorer-vocabulary-compatibility.md),
  would let a caller check that an external model was built for this
  vocabulary.
- **Bindings.** Rust is the first consumer (lwpc).
  `OfflineRecognizerResult` in `sherpa-onnx/rust/sherpa-onnx/src/offline_asr.rs`
  drops every JSON field except text, tokens, timestamps and durations,
  so `lang`, `lang_probs` and `ys_log_probs` are lost there today; fix
  that in the same change. Other bindings follow the fork's usual
  coverage ([fork-feature-bindings](../fork-feature-bindings.md)).
- **Upstream.** `token_ids` and the table are general and small; offer
  them upstream rather than carrying them as fork-only.

## Thread pools, secondary

A caller running its own ONNX model beside the recognizer, such as a
punctuation and case tagger, would like to share sherpa's threads rather
than add its own. Today each sherpa session sets its own intra-op and
inter-op thread counts (`GetSessionOptionsImpl` in
`sherpa-onnx/csrc/session.cc`), so every session has a private pool;
there is no shared pool to hand out. Sharing would mean ONNX Runtime's
global thread pools (an environment created with global thread pools and
sessions that disable per-session threads), chosen by the process, with
sherpa accepting that choice. It is independent of the token API.

On the phone, where sessions run with one thread (graehl, 2026-11-19),
this is moot: ONNX Runtime with one intra-op thread computes on the
calling thread and keeps no pool threads, and the default sequential
execution mode never uses the inter-op pool. There is nothing to share;
the caller schedules the tagger on whichever thread it likes. Sharing
matters only for multi-threaded server use.

What can be shared at one thread is memory for intermediate tensors.
Each session's CPU arena grows to that session's peak and keeps it, so
the ASR model and a tagger run one after the other on the same thread
hold the sum of their peaks. Two ways to get nearer the larger one, both
reachable without changing sherpa, since a provider string of the form
`cpu:<file>` reads ONNX Runtime settings from a file
(`ParseConfigFile` in `session.cc`), untried here:

- `EnableCpuMemArena=0` in that file, and no arena on the tagger's
  session, so freed tensors go back to the system allocator.
- One arena for both: the caller registers a CPU allocator on the ONNX
  Runtime environment before creating the recognizer, and both sessions
  set `session.use_env_allocators=1` (for sherpa,
  `SessionConfig.session.use_env_allocators=1` in the file). This relies
  on ONNX Runtime keeping one environment per process, so that sherpa's
  environment objects and the caller's are the same one.

The tagger's intermediate tensors are a few megabytes against the
encoder's, so measure resident memory before choosing either.

With a LiteRT encoder (today only w2v-BERT CTC,
`offline-w2v-bert-ctc-litert-model.cc`), nothing is reachable from
outside: each model creates its own `LiteRtEnvironment`, passing only
the dispatch library directory, and allocates its tensor buffers per
fixed-length signature at load. Sharing would mean the caller creates
the environment, or the device context it wraps, and sherpa accepts it
through the C API as an opaque handle. LiteRT 2.3's environment options
already take a caller's OpenCL context and queue, EGL display and
context, Vulkan, Metal or WebGPU device; for the NPU only the dispatch
library is passed. A tagger on ONNX Runtime shares nothing with a LiteRT
encoder except the process heap. Sherpa also sets no CPU thread count
for LiteRT, so the one-thread case is unchecked there.

## Not wanted

A callback inside sherpa that rewrites text. For offline recognition
nothing in sherpa runs after the rewrite, so a callback gains nothing
over the caller rewriting the result, and it would run caller code on
sherpa's decode thread.

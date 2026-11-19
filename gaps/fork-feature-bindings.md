---
slug: fork-feature-bindings
noticed: 2026-11-13
where: sherpa-onnx/c-api and every language binding built on it or on the core
---

# What the language bindings' checks did not cover

**Gap:** Every binding exposes the fork's options (FORK.md): fusion and
`options_json` on both recognizer configs, the w2v-BERT model, LiteRT and
encoder-window settings, and encode methods, with `literal_zeros` letting
bindings that fill in their own defaults send an exact 0. Each binding that
can run here decoded a real model through its own API: prefix search
matching greedy, a typed KenLM scorer, an `options_json` override that
changed the transcript, a rejected option failing creation, encoder windows
changing a word after encoding first, and a streaming transducer with a
scorer. These were not run:

- **Swift**: not compiled; checked by reading against `c-api.h`. With
  `SWIFT_PACKAGE`, `Package.swift` pulls upstream's v2.0.1 binaries, whose
  header lacks the new members, so a Swift package build fails until the
  binaries are built from this fork.
- **HarmonyOS ArkTS**: checked by reading. Its native code is the Node.js
  addon's, which ran.
- **WebAssembly**: the packer was checked against clang's wasm32 struct
  layout with a fake heap, at every binding commit; nothing was built with
  emscripten, so the exported encode functions were never linked.
- **Flutter**: not built; the Dart FFI layer ran under the Dart SDK, and the
  Flutter web wrapper was checked against the WebAssembly packer's keys by
  reading.
- **Android apps**: their Kotlin API files compiled with stub Android
  classes; the apps were not built.
- **w2v-BERT and LiteRT fields**: set and printed through bindings, never
  run with such a model.
- **Intermediate commits of the C-struct mirrors** were checked for layout
  against that commit's `c-api.h`, not run.
- **The waiting-list and stats fields** (`trace_survival`, `prune_waiting`,
  the waiting limits, a scorer's `stats`; 2026-11-19) ran through the C
  API, Python and JNI with a TDT model; Go and Rust were type-checked, the
  Node native code compiled and the WebAssembly packer checked on a fake
  heap against clang's wasm32 layout. C#, Pascal, Swift, Dart, Kotlin and
  ArkTS were checked by reading only: no toolchain for them was at hand.

Known sharp edges, documented where they bite:

- Go leaves `LiteralZeros` clear, so a 0 there is the default; with it set,
  every numeric field must be filled in.
- A C# `ScorerConfig` made without its constructor (`default`, a new array)
  sends weight and threads 0 under `literal_zeros`.

Also seen, upstream's and not fixed here: HarmonyOS's
`OfflineModelConfig.telespeechCtc` does not match the `teleSpeechCtc` key
its native code reads, and it has no `cohereTranscribe` field.

Scoped out: the multi-utterance decoder (see
[a public inflight decoder API](public-inflight-decoder-api.md)), and the
bulk program's batching and threads, which are command-line orchestration.

**Closure:** build Swift, HarmonyOS, WebAssembly and Flutter and run the
check above in each, or record per language why it cannot be run.

Contributing-model: fable-5.2
Contributing-model: 6-Astra
Contributing-model: opus-5.5

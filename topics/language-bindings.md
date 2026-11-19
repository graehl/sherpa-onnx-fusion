# Language bindings for the fork's options

> How the fork's options and methods reach every language binding: typed
> fields in the C API and in each binding, a JSON object of command-line
> options applied after them, a flag for exact zeros, and the rule that a new
> option is mirrored everywhere in the commit that adds it or the next.

Topic: language-bindings

## The contract

- **Typed fields.** The C API (`sherpa-onnx/c-api/c-api.h`) carries the
  fork's options as structs: `SherpaOnnxFusionConfig` (a pointer and count
  of `SherpaOnnxScorerConfig`, plus the prefix-search settings) as `fusion`
  on both recognizer configs; the w2v-BERT model, LiteRT settings and
  `SherpaOnnxOfflineEncoderConfig` on the offline model config. Members are
  appended, never inserted, so a binding that mirrors the layout appends in
  the same order. The conversion to core configs lives in
  `sherpa-onnx/c-api/c-api-fusion.cc`.
- **Zero is the default in C.** As elsewhere in the C API, a zero or NULL
  field means the core default, so a zero-filled struct works.
  `literal_zeros` on `SherpaOnnxFusionConfig` and on
  `SherpaOnnxOfflineEncoderWindowConfig` makes a 0 in their numeric fields
  (and the scorers') mean 0. A binding that carries the real defaults sets
  it, so 0 means the same as on the command line; one that cannot tell a
  set 0 from an unset field (Go) leaves it clear. A flag whose default is
  on (`prune_waiting`, a scorer's `stats`) follows the same rule: 0 is the
  default unless `literal_zeros` is set, when it turns the flag off.
- **`options_json`.** Both recognizer configs take a JSON object of
  command-line options, applied after every typed field through
  `ApplyOptionsJson` (`sherpa-onnx/csrc/recognizer-options-json.h`, over
  [option collections](json-option-collections.md)). Anything the typed
  fields lack needs no new member; an error in it fails creation. Python
  and JNI, which bind the core directly, apply it the same way.
- **Methods.** Offline recognizers have encode methods beside their
  decode methods (`SherpaOnnxEncodeOfflineStream`,
  `SherpaOnnxEncodeMultipleOfflineStreams`).
- **Lifetimes.** A stream may outlive its recognizer: a streaming search
  keeps the scorers it uses alive.

## Where each language lives

| Binding | Over | Files |
|---|---|---|
| C++ wrapper | C | `sherpa-onnx/c-api/cxx-api.{h,cc}` |
| Python | core | `sherpa-onnx/python/csrc/`, `sherpa-onnx/python/sherpa_onnx/` |
| Java, Kotlin | core (JNI) | `sherpa-onnx/jni/`, `sherpa-onnx/java-api/`, `sherpa-onnx/kotlin-api/` (linked into the Android apps) |
| Go | C | `scripts/go/` |
| Rust | C | `sherpa-onnx/rust/` (sys and safe crates) |
| C# | C | `scripts/dotnet/` |
| Pascal | C | `sherpa-onnx/pascal-api/` |
| Swift | C | `swift-api-examples/SherpaOnnx.swift` |
| Node.js, HarmonyOS | C | `harmony-os/SherpaOnnxHar/sherpa_onnx/src/main/cpp/` (shared native code), `scripts/node-addon-api/lib/`, the ArkTS sources |
| WebAssembly | C | `wasm/asr/sherpa-onnx-asr.js` (struct packing by hand) and the size asserts in `wasm/*/` |
| Dart, Flutter | C | `flutter/sherpa_onnx/lib/src/` (FFI structs, config classes, web wrapper) |

## Adding an option

A new option of the fork goes into the C struct, its conversion, the C++
wrapper and every binding above, in the commit that adds the option or the
one after it, with the binding's real default. A binding that mirrors the C
layout must append the member exactly where `c-api.h` does. Check it
through the binding: a nondefault value must reach the core (the core
prints its config with `debug`) and change what it decodes.

What the bindings' checks have not covered (Swift, HarmonyOS ArkTS,
WebAssembly and Flutter builds) is in
[the bindings gap](../gaps/fork-feature-bindings.md); a public API for the
multi-utterance decoder is
[its own gap](../gaps/public-inflight-decoder-api.md).

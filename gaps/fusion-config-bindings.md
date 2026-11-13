---
slug: fusion-config-bindings
noticed: 2026-11-13
where: OfflineRecognizerConfig::fusion (sherpa-onnx/csrc/offline-recognizer.h) and every language binding
---

**Gap:** `OfflineRecognizerConfig` has a `fusion` member (scorers, length
bonus, CTC prefix-search options) that only the C++ struct and the command
line expose. The C API, Python, JNI, and the other bindings neither set it
nor pass it through, so `decoding_method = "prefix_beam_search"` from a
binding runs without scorers and with default search options.

**Noticed while:** Wiring the CTC prefix-search decoder into the offline
recognizer. The project's convention is that a new config field is mirrored
in each binding.

**Fix sketch:** A list of scorer structs is awkward in the C API. Consider
passing the fusion configuration to bindings as one JSON string, applied
with `ApplyJsonOptions`, instead of mirroring every field in every
language; decide before the first binding is touched. Keep `Validate()`
as the single place that rejects scorers given with another decoding
method.

Contributing-model: fable-5.2

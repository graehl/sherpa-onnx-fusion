---
slug: upstream-ci-toolchain-drift
noticed: 2026-11-18
where: flutter-examples/*/android, .github/workflows/test-flutter.yaml, cmake/onnxruntime-win-x64-gpu.cmake
---

# Inherited build parts that outside changes broke

These come from upstream k2-fsa/sherpa-onnx, not from the fork; CI was
patched around them on 2026-11-18 rather than fixed.

- **Flutter examples on Gradle 8.14.** Flutter 3.50 (stable since
  2026-11-09) refuses Gradle older than 9.3.1. Every example's `android/`
  project pins Gradle 8.14 and AGP 8.11, so `test-flutter`'s Android jobs
  were pinned to Flutter 3.47.9. The lasting fix is moving the examples to
  Gradle 9 (and whatever AGP and Kotlin that needs), then unpinning; the
  six disabled `test-flutter-*` workflows would need the same.
- **ONNX Runtime downloads from github.com/microsoft/onnxruntime.** That
  repository answered 404 on 2026-11-18. `test-onnxruntime-version` moved
  to the csukuangfj/onnxruntime-libs mirror (dropping 1.17.1, which the
  mirror has only as a static library), but
  `cmake/onnxruntime-win-x64-gpu.cmake` still downloads from it, so a
  Windows build with `SHERPA_ONNX_ENABLE_GPU=ON` cannot fetch ONNX Runtime.
  Both workflows that build it are disabled.

## Closure

The examples build with current Flutter unpinned, and no build file
downloads from a URL that does not answer.

Contributing-model: claude-opus-5-5

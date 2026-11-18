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
- **Go's `non-streaming-tts-phonemize` on Windows x86.** It links against
  the third-party `github.com/csukuangfj/piper-phonemize-go-windows`
  v1.5.2, whose 32-bit DLLs lack the `PiperPhonemize*` symbols its cgo
  code calls, so `test-go` group 3 on windows-2022 x86 fails to link.
  Upstream's run of 2026-11-13 fails the same job.
- **Termux i686.** Both i686 jobs of `Test rust` failed on 2026-11-18
  because `apt` in a freshly pulled `termux/termux-docker:i686` image could
  not resolve `packages-cf.termux.dev`, which the other architectures'
  images reached minutes earlier. Unconfirmed whether it recurs.

## Closure

The examples build with current Flutter unpinned, and no build file
downloads from a URL that does not answer.

Contributing-model: claude-opus-5-5

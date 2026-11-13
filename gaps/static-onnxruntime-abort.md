---
slug: static-onnxruntime-abort
noticed: 2026-11-13
where: default static build (cmake/onnxruntime-linux-x86_64-static.cmake) with GCC 15
---

**Gap:** With the default configuration on Linux x86-64 and GCC 15.2, every
binary that loads a model aborts with `free(): invalid pointer` while
creating the ONNX Runtime environment. The backtrace is inside the
prebuilt static `libonnxruntime.a`, in a `std::regex` constructor called
from `onnxruntime::ParseSemVerVersion`, reached from `OrtApis::CreateEnv`.
A build with `-DBUILD_SHARED_LIBS=ON`, which uses the shared ONNX Runtime,
works. Unit tests that do not create an ONNX Runtime environment pass in
both.

**Noticed while:** Running `sherpa-onnx-offline` end to end to check the
prefix beam search wiring. Greedy search aborts the same way, so the cause
is not in this fork's changes; whether upstream at the same commit aborts
on this compiler was not checked.

**Fix sketch:** Unknown; needs diagnosis. Likely the prebuilt static
library and the program each carry an incompatible copy of libstdc++ state
(the build links with `-static-libstdc++`). Try without
`SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY`, and a newer ONNX Runtime static
package.

Contributing-model: fable-5.2

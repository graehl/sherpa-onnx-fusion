---
slug: streaming-ctc-traceback-recomputation
noticed: 2026-11-12
where: sherpa-onnx/csrc/online-ctc-fst-decoder.cc — DecodeOne
---

**Gap:** Streaming FST CTC decoding repeatedly reconstructs the full
best-path output after each chunk for which `ReachedFinal()` succeeds.
Kaldi preserves search tokens and backpointers across `AdvanceDecoding()`
calls; the redundant work is traceback and output construction, not
repeated acoustic scoring or beam expansion. Runtime impact has not been
benchmarked.

**Noticed while:** Investigating the trailing-blank overcount captured by
`sherpa-onnx/csrc/online-ctc-fst-decoder-test.cc`. This performance issue is
separate from that counter's correctness defect, tracked in
[the overcount gap](streaming-ctc-trailing-blank-overcount.md).

## Evidence

- `cmake/kaldi-decoder.cmake` pins kaldi-decoder v0.4.0. Its
  `FasterDecoder::Decode()` calls `InitDecoding()` and `AdvanceDecoding()`;
  incremental search uses the same frame-processing loop.
- `FasterDecoder::GetBestPath()` in the dependency's
  `kaldi-decoder/csrc/faster-decoder.cc` follows the selected token's
  `prev_` chain, builds a reverse arc vector, constructs a linear FST, and
  removes local epsilons on each request.
- Sherpa's `DecodeOne()` then extracts complete symbol sequences, scans
  them, and replaces the token, word, and timestamp vectors. At fixed chunk
  size, repeated full-history walks can accumulate quadratic work in
  utterance length, even when the winning history remains unchanged.

## Fix sketch

Reuse path-derived state across result requests without assuming that the
previous winning prefix remains the winner. Committed prefixes are one
option, not a prerequisite or the only approach (user-directed).

Ordinary memoization keyed by stable token/path identity could cache shared
ancestry and derived CTC state. A persistent linked-list/backpointer result
representation exposed by `GetBestPath()` could also let successive results
share history, materializing flat arrays only when a consumer needs them.
Define ownership so cached nodes survive decoder pruning safely, and reclaim
unreferenced history on pruning/reset. Preserve word labels, timestamps,
repeat/blank collapse, final weights, and revised winning paths.

If the external API requires a fresh complete flat transcript every chunk,
copying that transcript still costs its output size. Separate this required
materialization from avoidable intermediate FST construction and repeated
derivation; moving the same full walk elsewhere does not resolve the gap.

## Closure checks

Compare outputs against the existing implementation on multiple chunkings,
including a later chunk that changes the winning earlier path, blank/repeat
boundaries, epsilon arcs, final-state selection, and reset. Measure traceback
work, allocations, runtime, and retained memory as utterance length grows,
separately from search and any requested full-output serialization.

Contributing-model: gpt-6-astra

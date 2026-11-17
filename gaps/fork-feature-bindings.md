---
slug: fork-feature-bindings
noticed: 2026-11-13
where: fork additions in sherpa-onnx/csrc and all non-C++ language bindings
---

# Language bindings for the fork's features

**Gap:** The fork's new capabilities mostly have C++ and command-line
interfaces only. Track every new fork feature here until its applicable
language APIs expose it, or explicitly document why no binding is needed.
This is a fork-wide inventory to complete, not a claim that every added
behavior requires a new API: internal fixes and model behavior may already
be reachable through existing bindings.

Known omissions and surfaces to audit:

- Fusion configuration: `OfflineRecognizerConfig::fusion` and
  `OnlineRecognizerConfig::fusion`, including scorer collections, weights,
  length bonuses and prefix-search controls. The C and Python configuration
  surfaces omit these members; selecting `prefix_beam_search` alone cannot
  configure the scorers or nondefault search options.
- Chunked encoding: `OfflineModelConfig::encoder`, including windowing,
  normalization, blending, allowed lengths and padding.
- LiteRT configuration: `litert_dispatch_dir`, `litert_window`, and access
  to fixed-length models and signatures through each binding's model setup.
- Separate encoding and search: `OfflineRecognizer::EncodeStreams` and
  the lifecycle needed to consume its outputs through ordinary decoding.
- Inflight decoding: determine the public API needed for admission,
  progress, completion and cancellation; do not expose internal classes
  merely to mirror C++ implementation details.
- Bulk-driver capabilities: length sorting, batch limits, separate search
  threads and attributable results. Distinguish reusable library controls
  from CLI orchestration; an executable's output formatting does not itself
  require a binding.

Use [FORK.md](../FORK.md) and the diff from upstream as the inventory,
including future fork additions. Audit C, Python, JNI/Java/Kotlin, and
the wrappers built on them, including Rust, Swift, .NET, Go, Node,
Dart/Flutter, Pascal, HarmonyOS and WebAssembly. The list above does not
certify that every binding has been inspected.

**Noticed while:** Wiring prefix search, then documenting encoder windows.
User-directed scope expansion on 2026-11-17: cover all new fork features,
rather than filing a separate gap for each option group.

**Fix sketch:** First build a feature-by-binding inventory with implemented,
missing and not-applicable entries. Decide once how nested options and
scorer collections cross the C boundary: typed structures, a JSON entry
point using `ApplyJsonOptions`, or a combination. JSON remains a proposal,
not a settled interface. Reuse core parsing and `Validate()` rather than
duplicating configuration semantics in each wrapper; preserve public ABI
and existing defaults. Configuration transport alone does not expose
execution methods or lifecycle operations.

Close each applicable entry with a real binding-level check showing that
nondefault options reach the core and affect execution, plus an example
and consistent error propagation. Keep this gap until the inventory is
complete and its omissions are resolved or explicitly scoped out.

Contributing-model: fable-5.2
Contributing-model: 6-Astra

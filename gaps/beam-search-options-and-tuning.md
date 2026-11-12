---
slug: beam-search-options-and-tuning
noticed: 2026-11-12
where: OnlineRecognizerConfig, OfflineRecognizerConfig, OnlineCtcFstDecoderConfig, OfflineCtcFstDecoderConfig
---

**Gap:** Beam-search controls are split across recognizer and decoder
configs, with inconsistent naming and incomplete exposure of consequential
settings. CTC FST search inherits cost thresholds without documentation
establishing their suitability for the supplied model and graph.

**Noticed while:** Comparing Kaldi FasterDecoder's pruning policy with
Sherpa's CTC and transducer configuration surfaces.

## Current behavior

- CTC FST configs expose `graph` and `max_active` (default 3000), through
  online `--ctc-graph` / `--ctc-max-active` and offline
  `--ctc.graph` / `--ctc.max-active`. Online and offline implementations
  construct Kaldi options and override only `max_active`.
- The pinned kaldi-decoder v0.4.0 therefore supplies `beam=16`,
  `min_active=20`, `beam_delta=0.5`, and `hash_ratio=2` implicitly.
- Transducer modified beam search uses recognizer-level `max_active_paths`
  (default 4), exposed as `--max-active-paths`. Its search implementation
  is separate from Kaldi FST search.
- The shared CTC FST runtime passes model scores and loaded graph weights
  through without additional acoustic scaling or temperature calibration.
  Scaling may be embedded in a model or graph; there is no common
  calibration contract making a cost beam of 16 comparable across them.

Relevant sources are `sherpa-onnx/csrc/online-recognizer.h`,
`sherpa-onnx/csrc/offline-recognizer.h`, and the online/offline
`ctc-fst-decoder-config` and `ctc-fst-decoder` files under
`sherpa-onnx/csrc/`.

## Requested direction

User-directed, 2026-11-12:

- Introduce a common beam-search options struct for genuinely shared
  controls, with decoder-specific options through composition or
  inheritance. Both CTC FST and transducer search have a maximum-active
  beam-width concept; give that concept a shared configuration home.
- Use the shared meaning: maximum active search hypotheses. A surviving
  token at an FST state is a hypothesis too. Document decoder-specific
  enforcement: Kaldi limits current states selected for expansion, whereas
  transducer top-K limits the selected candidate set. Kaldi's setting is
  not a hard cap on successor allocations or states after epsilon expansion.
- Keep decoder-specific defaults safe and sensible for their decoder.
  A shared struct need not impose one numeric default on every search:
  the existing 3000-state and 4-hypothesis defaults are not interchangeable.
  Choose and justify defaults rather than merely inheriting library values.
- Document high-payoff or necessary model/platform-specific tuning both
  beside the relevant fields in code and in a committed repository topic.
  In particular, explain the units and score-scaling assumptions behind
  the CTC cost beam of 16, what can invalidate it, and how to tune it.

## Implementation and documentation requirements

Determine which controls belong in the common struct and which are specific
to a search algorithm. Composition versus inheritance remains open. Do not
add meaningless settings to greedy decoding or pretend every beam search
supports every pruning policy.

Expose consequential CTC FST controls currently hidden by the wrapper.
Document cost-beam versus count-beam behavior, min/max-active interaction,
and `beam_delta` as slack for early successor rejection. Separate search
quality controls from implementation controls such as hash sizing. Explain
which settings depend on model score distributions, graph/LM weighting,
frame rate, or platform memory/latency budgets. Include a reproducible
tuning procedure and the evidence supporting recommended defaults;
unmeasured defaults must remain identified as such.

Keep existing public C++/binding/CLI contracts working through explicit
mapping or compatibility adapters. Define precedence if old and new options
can both be supplied. Sweep online/offline paths and affected language
bindings so the common configuration reaches the actual search consistently.

## Closure checks

Retain focused tests showing common and decoder-specific options reach
the real decoder and change the intended pruning behavior. Verify legacy
configuration compatibility, defaults, and conflicting-option handling.
Compare search quality and cost under representative model/graph and
platform conditions before claiming defaults are tuned. Land the code
documentation and canonical repository topic with the implementation;
remove this gap when those requirements are met.

Contributing-model: gpt-6-astra

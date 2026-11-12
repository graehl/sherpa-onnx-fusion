---
slug: executable-driver-infrastructure
noticed: 2026-11-12
where: sherpa-onnx/csrc executable drivers and WebSocket server implementations
---

**Gap:** Executable drivers share recognition and configuration libraries,
but duplicate application infrastructure without a consistent structure for
startup, resource ownership, transport, scheduling, and result delivery.
Common driver changes therefore require parallel edits, particularly in the
online and offline WebSocket servers.

**Noticed while:** Reviewing how file processors, microphone applications,
and servers expose shared beam-search options. User requested a gap for
missed code reuse and formal driver structure, 2026-11-12.

## Existing reuse and concrete duplication

- Recognizer/stream APIs already own model loading, feature extraction,
  and decoding. Recursive config registration shares CLI flags through
  `ParseOptions`. Keep these boundaries.
- File drivers already share `ReadWave`; microphone drivers share the
  `Microphone` PortAudio wrapper. Extend existing owners where appropriate
  instead of introducing parallel utilities.
- `sherpa-onnx/csrc/online-websocket-server.cc` and
  `sherpa-onnx/csrc/offline-websocket-server.cc` independently set up
  connection/work Asio contexts, thread-count options, and worker threads.
- `sherpa-onnx/csrc/online-websocket-server-impl.cc` and
  `sherpa-onnx/csrc/offline-websocket-server-impl.cc` repeat WebSocket
  initialization, open/close/message handler registration, logging setup,
  and parts of connection bookkeeping.
- The file and microphone entrypoints also own startup, input-loop, and
  reporting orchestration. Review those for worthwhile extraction rather
  than assuming that similar-looking loops have the same contract.

## Intended structure

Keep entrypoints small: compose configuration, input/transport, recognition
scheduling, and output/lifecycle components with explicit ownership. Start
with the demonstrated common server-runtime code; extract cohesive pieces
whose behavior actually matches, not a universal driver base class by default.

Separate shared mechanism from use-specific policy:

- Connection/work executors, thread lifetime, common transport setup, and
  logging can have shared owners where their contracts match.
- Streaming readiness, endpoint/reset handling, and partial-result delivery
  remain distinct from whole-utterance admission and batching. Streaming
  versus offline recognition is also independent of file, microphone, or
  network input.
- Preserve the existing online/offline WebSocket wire protocols. Shared
  transport infrastructure does not require one message parser or scheduler.
- Keep model/search options in the existing recognizer configuration layer;
  see [beam-search-options-and-tuning.md](beam-search-options-and-tuning.md).

Document the resulting component boundaries and extension points in the
repository so adding a driver has an established composition pattern. Decide
composition versus inheritance from the actual ownership and lifecycle
contracts. Avoid replacing a few repeated statements with a framework whose
mode flags scatter those same differences across more files.

## Closure checks

Identify which duplicated blocks were removed and which remain deliberately
separate. Preserve executable names, CLI flags/defaults, result formats, and
network protocols unless a compatibility change is explicitly agreed.
Exercise file recognition, microphone lifecycle where hardware is available,
and both server paths through startup, input completion, result delivery,
disconnect, and shutdown. Cover concurrent clients and stream ownership at
the extracted scheduling/transport boundaries; report unavailable checks.
Keep refactoring separate from decoder algorithm or search-policy changes.

Contributing-model: gpt-6-astra

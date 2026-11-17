---
slug: public-inflight-decoder-api
noticed: 2026-11-17
where: sherpa-onnx/csrc/ctc-inflight-decoder.h, the recognizer facades, and every language binding
---

# A public inflight decoder API

**Gap:** Inflight batching exists only as `CtcInflightDecoder`, an internal
class over CTC log probabilities. No recognizer facade, C API function or
language binding lets a caller admit an utterance, feed it, finish or
cancel it, and collect its result while other utterances stay active in
the same decoder. Binding that class directly would expose an
implementation detail, tie the interface to CTC, and leave the model run
to the caller.

**Intended scope:** one public API for every decoder that can keep
independent per-utterance search state between calls, not only CTC: the
CTC prefix search, the transducer and NeMo RNN-T/TDT prefix searches, and
any other search that batches its model or scorer calls across
utterances. A decoder that cannot take part should say so through the API
(a capability query, or a creation error), not silently decode one
utterance at a time.

**Fix sketch:**

- A facade beside `OfflineRecognizer`, created from the same config, that
  owns the model run as well as the search: callers give audio or features,
  never log probabilities, so a model family's encoder batching and its
  search batching stay internal.
- Operations by stable utterance handle: admit (with per-utterance
  options such as scorer weights), accept input, mark input finished,
  cancel, step or run until progress, status, partial and final results,
  and release. Back-pressure is reported, not thrown: admission and input
  can be refused when capacity or queued-input limits are reached.
- One coordinator calls it serially, as `CtcInflightDecoder` requires;
  thread use stays inside the implementation.
- A C API over handles and status codes that the other bindings wrap,
  with errors returned rather than thrown across the boundary.

The interface should not need to change when the model run later joins
the inflight schedule
([encoder-inclusive inflight scheduling](sketches/encoder-decoder-inflight-scheduling.md)).

**Noticed while:** Adding language bindings for the fork's features;
user-directed on 2026-11-17, scoped to all decoders that can do it.

Contributing-model: opus-5.5

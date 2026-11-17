---
slug: recurrent-scorer-states-on-host
noticed: 2026-11-17
where: sherpa-onnx/csrc/recurrent-lm-scorer.cc (Node, Compute), onnx-recurrent-lm-step.cc
---

# A recurrent scorer's states stay on the host when its model runs on a GPU

**Gap:** The fusion topic's execution placement says that backend-owned
neural states can remain on their accelerator
([beam-search-and-scorer-fusion](../topics/beam-search-and-scorer-fusion.md#execution-placement-and-batching)).
The onnx-rnn scorer keeps every node's recurrent state as a host
`std::vector<float>` whatever its provider. With
`--scorer.N.provider=cuda`, each step therefore copies every parent's
state from host vectors into the step's input and every new state back,
besides the next-unit log probabilities the search needs on the host
anyway. No commit or doc said so; the LSTM-on-GPU timings in the topic
were taken with this round trip.

The states themselves are computed incrementally: one model step per
prefix that something is scored after, from its parent's stored state.
The defect is only where they are kept.

**Cost:** not measured; it buys nothing in return. Per step it is a few
hidden-size vectors per hypothesis each way, more for icefall's
LibriSpeech LSTM than for the 256-unit test LSTM, and paid on top of the
cost of a change of shape that the step's varying width already incurs
on a GPU
([fixed-width scorer steps](../topics/beam-search-and-scorer-fusion.sketches.md#fixed-width-scorer-steps)).

**Fix sketch:** the search on the CPU never reads a state's contents; it
decides which nodes survive from their scores and holds nodes by handle.
So a state belongs where the model runs (graehl, 2026-11-17), and reading
it back serves nothing. Keep each computed node's state in device memory
(an `Ort::Value` on the provider's allocator, or rows of a pooled device
buffer), assemble a step's input by a gather on the device, and run the
step through ONNX Runtime IO binding so its state outputs stay there; only
log probabilities come back. Node release frees device rows. The CPU
provider keeps the host path, where host is where the model runs. The one
exception is device memory: live nodes' states then compete with the
models for it, and only a run on the edge of running out would justify
spilling states to the host.

Less than whole rows need come back (graehl, 2026-11-17: some pruning can
happen on the GPU first). The search reads only the units its extensions
name, but a node's row is cached for later extensions of the same prefix,
so the choices are a top-k row per node with a floor for the rest, or the
requested entries with a device lookup on a later miss. A token LSTM's
rows can be thousands wide, which makes this the larger transfer.

**Closure:** states that stay on the device a GPU scorer runs on, with
the step's time before and after recorded in the topic.

Contributing-model: opus-5.5

---
slug: transducer-search-batching
noticed: 2026-11-17
where: offline-transducer-prefix-beam-search-decoder.cc (DecodeTransducerFrames), nemo-transducer-prefix-segment.cc, the streaming transducer decoders
---

# Searching a batch of transducer utterances together

## What is missing

The CTC decoders step every utterance of a decode call in lockstep
(`DecodeFramesTogether`): each scorer wave gathers the extensions of all
live utterances into one `Advance`, so an LSTM scorer runs one model step
over hypotheses of different sentences. The transducer decoders search the
utterances of a call one after another: `DecodeTransducerFrames` drives one
`TransducerPrefixSearch` to its end, so each decoder and joiner call, and
each scorer `Advance`, holds one utterance's hypotheses. Only the
end-of-input scoring (`FinishTogether`) spans the batch. Streaming
transducers do the same across streams.

## Why it is worth doing

With an LSTM scorer the scorer's steps are plausibly a large share of the
search (graehl, 2026-11-17): on the English zipformer transducer the LSTM
took the real-time factor from 0.063 to 0.51, under heavy load (FORK.md).
Each step is a CPU ONNX call with a fixed cost, so steps over B utterances'
hypotheses at once should cost much less than B steps. The same holds for
the stateless decoder and joiner calls.

## Shape of the change

`TransducerPrefixSearch` already has the phased interface CTC's lockstep
uses: `Frontier`/`Frame`, `Expand`, `GatherScorerWave`/`ApplyScorerWave`,
`EndRound`, `EndFrame`. A `DecodeTransducerFramesTogether` would, per
round:

- run the decoder and joiner once over the concatenated frontiers of every
  live utterance, each row with its own utterance's encoder frame;
- `Expand` each search with its rows;
- run each scorer's wave once across all searches, as
  `RunScorerWavesTogether` does for CTC;
- continue only the searches whose `EndRound` asks for another round, and
  move a search to its next frame when its rounds end.

Searches advance at their own frames, so a round's model call mixes frames;
nothing requires them to be equal. A test like CTC's
`BatchMatchesDecodingEachUtterance` must show identical results to
decoding each utterance alone. NeMo models are a second step: the
prediction network's state is kept per hypothesis and already batched
within an utterance (`NeMoTransducerPrefixSegment`), and batching across
utterances concatenates more of the same.

## Shapes

On the CPU the varying width of these calls costs nothing measurable
([GPU decode cost](../gpu-decode-cost-outside-the-model.md)); on a GPU
each change of shape is expensive, so a GPU search would want the
fixed-width steps sketched for scorers in
[the fusion sketches](../../topics/beam-search-and-scorer-fusion.sketches.md#fixed-width-scorer-steps).

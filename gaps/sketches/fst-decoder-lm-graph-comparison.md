---
slug: fst-decoder-lm-graph-comparison
noticed: 2026-11-14
where: offline-ctc-fst-decoder, ctc-prefix-search, kaldi-decoder graphs
---

# Prefix search against the FST decoder on a graph with an LM

## Intended outcome

The [fusion topic](../../topics/beam-search-and-scorer-fusion.md#max-alignment-mode)
asks that max-alignment prefix search be compared with the Kaldi
`FasterDecoder` path before that path is retired. The comparison so far is
without a scorer, on a plain CTC topology graph: same result, far less
time, on a graph that is the worst case for an arc-walking decoder
(§ Matching the existing decoders). The comparison that matters gives both
decoders the same language model: `FasterDecoder` on a graph composed with
it, prefix search fusing it as a scorer, with error rate and time for
each.

User-directed (graehl, 2026-11-14): leave this an intended experiment
until a token-level LM contrast is finished and validated.

## Design notes

- **A graph from the character n-gram.** Compile the KenLM character
  model to a G acceptor over characters, with a lexicon L that spells each
  recognizer token as its characters and the CTC topology H, giving HLG.
  Backoff is the open question: the usual construction has one state per
  context and an epsilon backoff arc, which lets a path take the backoff
  even when the higher-order n-gram exists, so its scores are an upper
  bound on the model's and not the model's. Exact backoff needs failure
  arcs, which `FasterDecoder` does not follow. Either accept the
  approximation and report how often the two scores differ, or reduce the
  order until an exact expansion fits. An 8-gram is likely too large to
  compose; order 4 to 6 may be the practical limit.
- **A graph from a token n-gram.** A model over the recognizer's own
  tokens needs no lexicon (L is the identity), so the graph is H composed
  with G and should be much smaller. A CTC recognizer with a token LM is
  what this experiment needs. The prefix search fuses such a model with
  `--scorer.unit=token`.
- **What to hold equal.** Identical log probabilities to both decoders (the
  comparison program for the no-scorer case already does this), the same LM
  weight, and each decoder's beam chosen for equal error rate, then equal
  time. Report ties and score differences explicitly.

## Completion evidence

Error rate with significance and decode time for both decoders on one test
set with one LM, at matched settings, with the size of the graph and the
time to build it; and a statement of how exactly the graph's scores match
the model's.

Contributing-model: fable-5.2

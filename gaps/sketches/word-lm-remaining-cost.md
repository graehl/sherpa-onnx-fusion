---
slug: word-lm-remaining-cost
noticed: 2026-11-15
where: kenlm scorer word mode (`"words": true`), any scorer whose unit spans several recognizer tokens
---

# A remaining-cost estimate for partly spelled words

## Intended outcome

A word-level scorer under a subword recognizer charges nothing until a
word's boundary arrives, so inside a word the search prunes on the
recognizer's score alone. A decoding graph compiled from the same LM does
not have this blind spot: determinizing the lexicon with the LM leaves cost
on mid-word arcs (41% of the arcs without a word label carry weight in the
graph released with the public English zipformer CTC model), each paying as
soon as the tokens so far rule out the likely words.

The scorer could do the same with a stored estimate: for each prefix of a
vocabulary word, the best score any completion could get. Each token pays
the change in that estimate and the boundary pays the difference from the
true score, so finished words score exactly as now and only pruning
changes.

## Status

Deferred (graehl, 2026-11-15): the LM to use under a subword recognizer is
one over the recognizer's own tokens, which is scored token by token and
needs none of this. Wanted only if word-level LMs are to be used at small
beams.

## Design notes

- The estimate can ignore context (best unigram under the prefix, from a
  prefix tree built when the model loads) or use it (best completion given
  the LM state, cached). The graph's is the second kind.
- Increments can raise a score as well as lower it; see the notes on
  score-raising scorers in
  [the tuning sketch](fusion-parameter-tuning.md#a-parameter-to-include-subtracting-a-transcript-lm).
- One observation it might explain, untested: a pruned public word 3-gram
  scored 5.73 WER in word mode against 5.39 for the released graph on 600
  LibriSpeech test-other recordings, while an unpruned one scored 4.93. The
  graph also restricts output to its word list, uses its own LM scale, and
  may hold a different LM, so that gap has other candidate causes.
- A way to forbid out-of-vocabulary words would separate the lexicon
  constraint from the estimate.

## Completion evidence

With the estimate on, a word LM at a small beam gives the same result as
the same search with a wide beam on more recordings than without it, and
the pruned-LM gap above is explained or closed.

Contributing-model: fable-5.2

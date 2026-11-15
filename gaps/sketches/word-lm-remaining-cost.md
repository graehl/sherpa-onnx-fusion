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

## Two forms of the estimate

User-directed (graehl, 2026-11-15): keep the word mode, and sketch the
estimate in both forms.

- **Subword-wise, precomputed.** Spell every vocabulary word with the
  recognizer's tokenizer when the scorer is created and build a prefix tree
  over token sequences, each node holding the best score of any word
  below it. A token then moves one step in the tree. This matches what the
  recognizer can actually emit and costs one lookup per token, but it ties
  the scorer to one recognizer's token table, needs the tokenizer or a
  pronunciation-style list of each word's token spellings, and a word with
  several tokenizations needs all of them.
- **Character-wise.** Build the prefix tree over the words' characters.
  The scorer already receives characters, so it needs nothing from the
  recognizer and works with any token table, at the cost of one step per
  character and a tree with more nodes.

The character form fits the scorer interface as it is and is the one to try
first. Either way it is a niche need: KenLM, which is built for searches
whose unit is the word, has no such estimate, and a search over subword
tokens does better to use an LM over those tokens.

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

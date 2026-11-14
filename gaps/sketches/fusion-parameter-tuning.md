---
slug: fusion-parameter-tuning
noticed: 2026-11-14
where: scorer weights, length bonus, beams; reference-aware decode modes
---

# Tuning fusion weights and bonuses from the search's own evidence

## Intended outcome

Scorer weights and the length bonus are chosen today by decoding a
validation set once per candidate setting and keeping the best error rate.
That costs a decode per point, says nothing about why a setting is good,
and does not transfer between searches: a weight and bonus chosen with one
search rule produced long deletions with another on the same model and LM
(the topic's § Fusion on a transducer, and the token n-gram case recorded
there).

The intent is to tune from what the search can report about the reference
transcript, with the modes the topic already defines
([reference and oracle exploration modes](../../topics/beam-search-and-scorer-fusion.md#reference-and-oracle-exploration-modes)).
A reference is given to learn about the search and the objective; nothing
here supplies an answer in use.

## The idea

For each validation utterance the decoder reports, for its result and for
the reference, the components the ranking adds: the recognizer's log
probability, each scorer's raw score, and the length in the bonus unit.
The objective is linear in the weights and the bonus, so from those
components the score of the reference and of each retained hypothesis is
known for any weights without decoding again. Then:

- **Objective errors and search errors separate.** Rescore the reference
  and result with the same alignment procedure and sum or max objective.
  If the reference then scores above the result under the current weights,
  the search returned a lower-scoring output (a search error: widen a
  beam). If it scores below, the objective prefers the wrong output
  (change the weights or the bonus). Comparing a protected reference's
  alignment mass with only the result's beam-retained mass cannot make
  this distinction.
- **Weights are fit, not swept.** Choose weights and bonus that rank the
  reference, or the retained hypothesis nearest it, above the others on as
  many utterances as possible, by any method for linear models over n-best
  lists (minimum error rate training is the classical one). Decode again
  with the fitted setting, add the new hypotheses, and repeat until the
  lists stop changing, because a new setting brings new competitors.
- **Degenerate regions are visible before they are decoded.** A setting
  under which a much shorter output outscores the reference on some
  utterances predicts the deletion failure. The fit can carry a constraint
  against it.

## Mechanism the decoder needs

- **Honest mode:** the reference is observed and never changes expansion,
  ranking, or pruning. The search reports whether each reference prefix
  was still in the beam at each frame, and at the frame it left, the score
  gap and rank it would have needed. This is the survival trace
  ([score-gap beam sweep](../../topics/beam-search-and-scorer-fusion.md#score-gap-beam-sweep))
  applied to a path the search did not choose, where today it follows the
  result's own path.
- **Reference-preserving mode:** only when the reference's score is
  wanted and honest search pruned it. The reference's prefixes are
  protected from pruning, and the run is labelled an intervention. Its
  purpose is the score and margin of the reference, not a transcript.
- **Component reporting:** results already carry the recognizer's log
  probability and each scorer's raw score. The n-best list with those
  components, and the same for the reference, is the tuning input.
- **For CTC and for transducers:** score each fixed sequence with a
  constrained alignment procedure for that model family, using the
  configured sum or max objective. Protecting a prefix in a finite search
  does not establish its exact alignment score. Report any beam or other
  approximation used for this rescoring, separately from the scores
  retained by the ordinary search.

## Open questions

- Which loss to fit: error rate on the n-best, a margin between the
  reference and the best competitor, or the reference's rank.
- Whether beams should be tuned in the same loop, from the traced gaps and
  ranks, or after the weights are fixed.
- How settings transfer across search rules, models, and beam sizes, and
  whether a setting should be recorded with the search rule it was fit
  under.

## A parameter to include: subtracting a transcript LM

User-directed (graehl, 2026-11-14): when the recognizer's decoder is itself
LM-like, as an RNN-T's prediction network is, include in the deployment
sweeps the option of subtracting an estimate of the LM it learned from its
training transcripts while adding the external one. Upstream's form of
this is LODR, a bigram over the recognizer's tokens applied with a negative
scale inside `modified_beam_search`. graehl doubts that removing the
transcript-LM component helps and wants it measured in case it does.

Here it would be a second scorer over tokens with a negative weight, fit
together with the external LM's weight. The option parser accepts any
weight today; nothing has been run with a negative one, and a negative
weight makes a prefix's score rise as it is extended, which the beams and
the survival trace were not designed around. Check that first on a few
recordings. It applies to transducers; a CTC model has no such component.

## Completion evidence

On one model and scorer, weights and bonus fit from one or two decodes
match the error rate of the best swept setting on held-out data; the
report separates search errors from objective errors; and a setting that
produces deletions is flagged from the validation components before a
test decode shows it.

Contributing-model: fable-5.2

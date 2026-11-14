---
slug: scorer-vocabulary-compatibility
noticed: 2026-11-14
where: scorer creation, token-unit scorers, future per-request model choice
---

# Checking that a token scorer and a recognizer share a vocabulary

## Intended outcome

A scorer over the recognizer's tokens is bound to one recognizer's token
IDs when it is created: `kenlm` resolves the model's words against the
recognizer's token table by symbol, or takes the IDs with a constant
offset; `onnx-rnn` takes the IDs with an optional offset. Nothing checks
that the model and the recognizer mean the same tokens. A wrong pairing
does not fail; it scores nonsense.

Two things are wanted (graehl, 2026-11-14), neither built:

- **A compatibility check.** A digest of the token table (symbols in ID
  order, with the special tokens' roles) recorded with the scorer's model
  and compared when the scorer is created, so that an ID-based pairing is
  certified instead of trusted. Optional: models without a digest still
  load, with the offset stated in configuration.
- **One scorer for several recognizers.** Recognizers of different cost
  with the same vocabulary could share one scorer, chosen per request by an
  API or a server; the command-line programs have no per-request choice.
  Token tables found equal by digest would be treated as one vocabulary,
  so the scorer's token mapping is computed once and not copied per
  recognizer. Character scorers already do not depend on the recognizer,
  apart from the search's own token-to-character table.

## Design notes

- The mapping is fixed at scorer creation, after the recognizer's table is
  known. Resolving it lazily per request would allow a scorer created
  before its recognizers, at the cost of a lookup on a path that should
  not have one; a registry of vocabularies by digest gives the same
  freedom without that cost.
- The symbol-resolved form reads the token table a second time, briefly,
  and keeps only the resolved word of each token. Passing the recognizer's
  loaded table to the scorer factory would avoid the read; the factory
  interface takes only options today.
- Special tokens differ in role across toolkits (blank, unknown, sentence
  boundaries; KenLM reserves its own). A digest has to cover the roles the
  scorer relies on, not only the spellings.

## Completion evidence

A scorer whose recorded digest disagrees with the recognizer's table is
refused with a message naming both; two recognizers with equal tables
share one scorer in a test; a model without a digest still loads.

Contributing-model: fable-5.2

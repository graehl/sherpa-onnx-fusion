# Glossary

Terms this fork uses in a sense of its own. Upstream sherpa-onnx vocabulary
keeps its upstream meaning.

| term | definition | topic / refs |
|---|---|---|
| auxiliary alignment retention | Keeping a prefix that has left the hypothesis beam alive in the CTC recurrence, under its own budget, so its later emissions still add probability to surviving descendants. Planned for a later pass. | [sketches](topics/beam-search-and-scorer-fusion.sketches.md#auxiliary-alignment-retention) |
| beam-search-and-scorer-fusion | Efficient phone and server ASR inference with LM fusion, using shared search components, model-specific execution, and independently weighted scorers. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md) |
| characterization test | A test that passes while a known defect is present and fails, with instructions, once the defect is fixed. | [overcount gap](gaps/streaming-ctc-trailing-blank-overcount.md) |
| early partial-score beam | The pruning threshold applied to token extensions after only part of a scorer's charge is known, such as a character LM's first-character score. Separate from the hypothesis beam. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#proposed-adjustable-beam-for-partially-scored-expansions) |
| ER2 | A versioned, language-specific normalized error rate computed by a scorer that is not part of this repository; not plain WER. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md) |
| honest search | A decode in which a supplied reference is observed and scored but changes no expansion, ranking, pruning, or result. Contrasts with reference-preserving search. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#reference-and-oracle-exploration-modes) |
| hypothesis beam | The retained set of distinct output histories, limited by score gap and by minimum and maximum active counts. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#decode-options-and-pruning-direction) |
| inflight batching | Admitting, finishing, and cancelling individual utterances while others in the same decoder stay active, with ready work batched across them. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#required-inflight-batching) |
| json-option-collections | Configure a vector of option structs through indexed command-line paths such as `--scorer.1.weight`, and through JSON that maps onto the same paths, using each struct's ordinary `Register(ParseOptions*)`. | [json-option-collections](topics/json-option-collections.md) |
| max-alignment mode | CTC search that scores an output by its single best alignment and merges hypotheses whose scorer state bytes are equal and whose CTC tails match. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#max-alignment-mode) |
| scorer | A pluggable model that adds a log score to a hypothesis as its output grows: an LM, or anything with the same state-and-advance contract. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#pluggable-scorer-contract) |
| scorer state | A fixed number of bytes per hypothesis per scorer, owned by the decoder. The scorer declares the size and optional copy and destroy functions; an n-gram's bytes are its context, a recurrent LM's hold a reference to its own storage. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#state-and-scoring-operations) |
| scorer unit | The symbol a scorer consumes one at a time: a recognizer token, a character, or the scorer's own token. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#pluggable-scorer-contract) |
| sum-over-alignments mode | CTC prefix search that scores an output by the summed probability of all its alignments, keyed by token sequence. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#search-semantics) |
| survival-requirement trace | For one hypothesis, the widest value each pruning limit would have needed at any step to keep it. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#conceptual-search-policy-tuning-support) |
| utterance | One logical decoding input with its own search state, scorer context, weights, and results, identified by a stable handle independent of batch position. | [beam-search-and-scorer-fusion](topics/beam-search-and-scorer-fusion.md#draft-decoder-lifecycle-api) |

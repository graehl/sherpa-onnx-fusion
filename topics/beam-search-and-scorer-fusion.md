# Shared beam search and scorer fusion

> Efficient phone and server ASR inference with LM fusion, using shared
> search components, model-specific execution, and independently weighted scorers.

Topic: `beam-search-and-scorer-fusion`

## Direction and status

The goal is efficient phone and server ASR inference with LM fusion. Add
sum-over-alignments CTC fusion and improve the non-CTC use cases that already
have fusion support. Success means useful recognition quality at acceptable
latency, throughput, memory, and sustained power/thermal cost on the target
hardware. The interfaces and code-sharing choices below serve that goal.

The decoder is implemented in this fork for CTC, stateless-decoder
transducers, and NeMo RNN-T and TDT models; a contribution to Kaldi or
upstream can follow. Sections below say what each part does and name its
source file; the code is the authority for signatures. Text about parts
that are not built is marked as such, and the original pseudocode for
everything is kept in the
[sketches companion](beam-search-and-scorer-fusion.sketches.md).

### What is built

| Part | Source (under `sherpa-onnx/csrc/`) | Notes |
|---|---|---|
| Scorer interface, registry, options | `scorer.h`, `scorer-options.h` | § Scorer interface |
| Shared prefix trie with scorer waves | `scored-prefix-trie.h` | § Model and state representation |
| CTC prefix search, sum and max | `ctc-prefix-search.h` | § Search semantics |
| Transducer, RNN-T, TDT prefix search | `transducer-prefix-search.h` | § Search semantics |
| `kenlm` scorer | `kenlm-scorer.h` | needs `-DSHERPA_ONNX_ENABLE_KENLM=ON` (KenLM is LGPL) |
| `onnx-rnn` scorer (LSTM or GRU) | `recurrent-lm-scorer.h`, `onnx-recurrent-lm-step.cc` | character units |
| Options and JSON | `fusion-config.h`, `option-collection.h`, `option-json.h` | § Options; [option collections](json-option-collections.md) |
| Offline CTC recognizer | `offline-ctc-prefix-beam-search-decoder.h` | batch shares scorer calls |
| Model run ahead of the search, offline CTC | `OfflineRecognizer::EncodeStreams` | § Inflight batching |
| Streaming CTC recognizer | `online-ctc-prefix-beam-search-decoder.h` | search kept per stream |
| Offline transducer recognizers | `offline-transducer-prefix-beam-search-decoder.h` | stateless and NeMo |
| Multi-utterance CTC decoder | `ctc-inflight-decoder.h` | § Decoder lifecycle |
| Survival trace | `ctc-prefix-search.h` (`trace_survival`) | § Score-gap beam sweep |

All of it is selected with `--decoding-method=prefix_beam_search`. Unit
tests sit beside each file (`*-test.cc`); the searches are tested against
enumeration of every alignment or path.

### Not built

- Streaming transducer recognizers, and the recognizers' use of the
  multi-utterance decoder (they batch within one decode call).
- Token-unit scorers for `kenlm` and `onnx-rnn` (the search supports token
  units; the scorers lack the recognizer's token table), and the existing
  `OnlineRnnLM` as a scorer type.
- The early partial-score beam, `Next`-based candidate scoring, and the
  tuning traces other than survival.
- Batched prediction-network calls for NeMo transducers (one hypothesis at
  a time now), and scorer end-of-input calls batched across utterances.
- Byte-BPE token tables, join-marker conventions the recognizer does not
  itself remove, and text normalization between recognizer and scorer.
- The options in the language bindings
  ([gap](../gaps/fusion-config-bindings.md)).

### Scope across model families

Follow graehl's [upstream proposal, issue #4186](https://github.com/k2-fsa/sherpa-onnx/issues/4186):
a pluggable scorer interface, multiple scorers at once, and character or token
units per scorer. CTC prefix search is one consumer of that interface; the
proposal also covers transducer fusion and adapting the existing online RNN
LM. An explicit OpenFst graph must not be required to use this decoder.

The intended scope includes non-CTC models, with maximal code reuse where
it does not cost efficiency. Plan for model-specific execution loops sharing
search components, each loop free to schedule its own way.
Correctness-critical semantics come from the model or a policy bundled with
it. Virtual policy objects, templates, or model-associated options are
implementation choices; avoid unrelated switches that allow invalid semantic
combinations.

Share prefix storage, scorer fusion, compatible batching utilities,
candidate processing, pruning, and result handling. Model policies define expansion, progress, legal
recombination, score accumulation, and completion. CTC needs blank/nonblank
alignment sums; transducers distinguish acoustic-time and output advancement;
autoregressive decoders advance output and caches toward EOS. Reuse scheduling
components where useful without imposing one frame-synchronous loop.
Recognition-model state remains distinct from auxiliary scorer state.
Existing public decoder interfaces can wrap these implementations to preserve
upstream integration.

CTC consumes frame scores that can be computed across available encoder
outputs in advance. Its acoustic recurrence does not need to schedule a
hypothesis-dependent ASR-network call per frame; only auxiliary neural
scorers introduce that kind of dependency. Retain inflight batching without
imposing the other model families' neural scheduling machinery on this path.

RNN-T and TDT schedule prediction-network and joiner work conditioned on
active hypotheses. Reuse prediction outputs across blank/time advances when
the token history is unchanged. TDT's encoder outputs may already exist,
but the acoustic positions visited by each hypothesis depend on predicted
durations and are not known in advance. Its post-encoder work therefore
needs scheduling, potentially on GPU, even in offline recognition.
Transformer decoding likewise schedules neural steps and cache updates
conditioned on surviving token histories. These differences justify separate
execution loops without duplicating the shared scorer and search facilities.

The chosen abstract ASR-model structure must be batch- and thermal-efficient,
without sacrificing efficiency relative to model-specific decoders
(graehl, 2026-11-12). Code reuse alone does not satisfy this goal. Preserve
model-specific opportunities for ragged batching, state/cache reuse, fused
operations, and device-resident execution. The interface must not force
per-hypothesis calls, redundant tensor copies, CPU/GPU synchronization, or
one universal expansion schedule where specialized scheduling avoids them.
Allow policies to submit coarse batched work and retain backend-owned state;
specialize execution paths where necessary rather than imposing an inefficient
lowest-common-denominator operation on every model.

Validate this against model-specific decoding at matched search semantics,
quality settings, model precision, provider, and workload. Compare throughput,
latency, memory, and energy per audio duration, including sustained operation
under target-device thermal limits. Any remaining overhead must be measured
and explained; equivalent interfaces or nominal batch sizes do not establish
equivalent efficiency. If the abstraction causes a meaningful regression,
revise its boundary or execution policy rather than accepting that regression
as the cost of sharing code.

## Evidence: alignment summation and LM fusion

### Recognizer path

The agreement above feeds the search cached log probabilities. Through the
recognizers, with models loaded and no scorer, `prefix_beam_search` gave
the same transcripts as the existing decoding methods on the models' own
test recordings (2026-11-13, shared-library build, x86-64, int8 models):

| Model | Family | Recordings | Compared with | Result |
|---|---|---:|---|---|
| Omnilingual ASR CTC 1B | CTC | 3 | greedy | identical |
| NeMo Citrinet-512 CTC | CTC | 3 | greedy | identical |
| Zipformer transducer (Vietnamese) | stateless transducer | 1 | greedy, modified beam search | identical |
| GigaAM v3 RNN-T | NeMo RNN-T | 1 | modified beam search | identical |
| Parakeet TDT 0.6B v3 | NeMo TDT | 2 (English, German) | greedy, modified beam search | identical |

This shows the wiring and the three transducer topologies running on real
models; a handful of recordings says nothing about error rate, and no
scorer was fused in these runs. On the TDT model the prefix search took
about three times the greedy decoder's time (real-time factor 0.63
against 0.22, one run), because it calls the prediction network once per
hypothesis where a batched call would do (§ Not built).

### Score-gap beam sweep

| Clips | Scorer, weight, word bonus | Beam | Largest gap that changed a transcript | Changed at 8 | Changed at 10 | Kept, no gap | Kept at 12 | Kept at 10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Vietnamese | none | 8 | 2 | 0 | 0 | 8.0 | 8.0 | 8.0 |
| Vietnamese | 8-gram, 0.3, 2 | 4 | 4 | 0 | 0 | 4.0 | 4.0 | 4.0 |
| Vietnamese | 8-gram, 0.3, 2 | 8 | 2 | 0 | 0 | 8.0 | 8.0 | 7.8 |
| Vietnamese | 8-gram, 0.3, 2 | 32 | 8 | 1 | 0 | 31.9 | 30.0 | 26.7 |
| Sinhala | none | 8 | none down to 1 | 0 | 0 | 8.0 | 7.1 | 6.9 |
| Sinhala | 8-gram, 0.5, 0.5 | 8 | 3 | 0 | 0 | 8.0 | 6.9 | 6.6 |
| Sinhala | 8-gram, 0.5, 0.5 | 32 | 6 | 0 | 0 | 31.9 | 25.0 | 21.7 |
| Sinhala | LSTM int8, 0.5, 0.5 | 8 | 3 | 0 | 0 | 8.0 | 6.9 | 6.6 |

Every change at the gaps shown was one character in one clip. No
transcript changed at a gap of 10 or more, and 12 leaves a margin over
the one condition that changed at 8. At 12 the gap removes little work at
beams of 4 and 8 and 6 to 22 percent of kept hypotheses at beam 32. Forty
clips can show that a gap is too small; they cannot show that 12 is safe
in general, and the measure here is agreement with the unlimited search,
which is not error rate against references.

A constant is a first default. The right gap depends on how far apart
competing hypotheses' scores sit, which depends on the recognizer's score
scale and on each scorer's weight and range; a default derived from those
would hold across models where a constant will not; see the sketches
companion.

The survival trace (`CtcPrefixSearchOptions::trace_survival`) reports, for
each result, the score gap and the hypothesis-beam rank it needed. In max
mode it follows the result's one best alignment, per ending, the way a
backpointer would, and a search given the traced gap plus a small margin,
or the traced rank, reproduces the result and its score exactly; tests
check this with and without recombination. In sum mode it takes the most
demanding of every alignment that added to the output, which also
reproduces the result exactly but asks for more than a search needs: an
earlier, looser version of the trace reported gaps of 11.5 to 18.3 on
these clips while no transcript changed at a gap of 10, because a prefix
pruned on the first weak frame of a new token is created again on the next
with little lost. For choosing a default in sum mode a sweep remains the
measure.

### Sum versus max on a full test set

| Scorer | Beam | Score gap | ER2 sum | ER2 max | ER2 max, no recombination | Search s, sum | Search s, max | Kept per frame, sum | Kept per frame, max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 8 | none | 11.40 | 11.37 | 11.38 | 2.2 | 2.3 | 8.0 | 8.0 |
| none | 32 | none | 11.43 | 11.37 | 11.38 | 8.5 | 11.4 | 31.9 | 30.3 |
| 8-gram | 8 | none | 8.26 | 7.79 | 8.23 | 9.5 | 12.0 | 8.0 | 8.0 |
| 8-gram | 16 | none | 7.79 | 7.44 | 7.83 | 16.9 | 22.6 | 16.0 | 16.0 |
| 8-gram | 16 | 12 | 7.79 | 7.44 | 7.83 | 15.4 | 15.4 | 14.4 | 10.7 |
| 8-gram | 32 | none | 7.55 | 7.29 | 7.51 | 31.1 | 42.9 | 31.9 | 31.9 |
| 8-gram | 32 | 12 | 7.55 | 7.29 | 7.51 | 25.5 | 21.4 | 26.0 | 15.2 |
| LSTM | 8 | none | 7.96 | 7.88 | 7.88 | 332 | 335 | 8.0 | 8.0 |

With the n-gram, max mode is better than sum mode at every beam: by 0.48
at beam 8 (95% interval 0.34 to 0.63), 0.35 at 16 (0.22 to 0.50), and 0.26
at 32 (0.14 to 0.38), each p < 0.0001 by paired bootstrap over clips.
The gain is recombination's: max mode without it scores the same as sum
mode (differences of 0.03 and 0.04 at beams 8 and 32, p above 0.3). Summing alignments
therefore gained nothing measurable over the best alignment here; what
helped was a beam that holds different n-gram contexts where sum mode's
holds several prefixes the scorer no longer distinguishes. Max mode at
beam 8 equals sum mode at beam 16 (7.79) in 12.0 s against 16.9 s, and max
mode at beam 16 is at least as good as sum mode at beam 32 (7.44 against
7.55, p = 0.06) in 22.6 s against 31.1 s.

At an equal beam max mode with recombination takes 27 to 38 percent
longer than sum mode while scoring the same number of new prefixes.
The cause was not measured (the host does not permit profiling); the
likely one is that its beam's n-gram contexts are all different, so the
scorer's lookups repeat less. The score gap removes that cost: the
recombined beam's hypotheses are spread further apart in score, a gap of
12 keeps 15 of 32 where sum mode keeps 26, and no transcript changed.

With the LSTM nothing recombines, because its state depends on the whole
history, and the two modes do not differ (0.08, p = 0.25). Its search time
is the model's steps. Without a scorer, max mode is the greedy decoder;
the small differences in that row are ties in the cached probabilities
broken differently.

Limits: one language, one recognizer, one test set, and one weight and
bonus per scorer, neither tuned per mode; the LSTM's were not tuned at
all. Recombination for sum mode (adding the probabilities of hypotheses
with the same future) is the untested combination these results point at.

## Search semantics

CTC prefix beam search is the established name for the prefix-based
algorithm used here to approximate summation over CTC alignments. Prefix
identity is the emitted token sequence; different tokenizations that render
the same text are not automatically merged.

For CTC sum-over-alignments mode, keep blank-ending and nonblank-ending
acoustic probability sums for each
token prefix, in log space. Merge contributions with log-add before pruning
prefixes. Blank transitions and repeated-frame labels that emit no new
token must not advance a scorer or charge a length bonus. A repeated token
emitted after a blank is a genuine extension.

Keep acoustic mass separate from each scorer's subtotal. A prefix's ranking
combines its log-summed acoustic mass, the weighted scorer subtotals, and
the configured length bonus. Equal token prefixes with deterministic
prefix-dependent scorer state share those subtotals across alignments;
alignment multiplicity must not multiply LM charges. Beam pruning makes
the alignment sum approximate: a pruned prefix stops feeding later
emissions to its surviving descendants. A later pass adds a separately
budgeted
[auxiliary alignment retention](beam-search-and-scorer-fusion.sketches.md#auxiliary-alignment-retention)
to recover those alignments.

### Transducers

For transducer models the recognizer's probabilities depend on the
hypothesis, so `TransducerPrefixSearch` asks the caller to run the model
for its current hypotheses and takes their log probabilities. Three
topologies share the code: blank or one token and then the next frame
(the modified beam search used for stateless-decoder transducers); RNN-T,
where a token stays in the frame and blank advances; and
token-and-duration, where each transition also chooses how many frames to
advance. A prefix's probability is tracked by the frame it has reached, so
the same token sequence reaching the same frame by different paths adds
up, and TDT hypotheses at different frames coexist. Within a frame, at
most a fixed number of expansions happen and the last allows only
transitions that leave the frame; a TDT blank that would not advance is
treated as advancing one frame. Scorers, weights, length bonus, and both
beams work as for CTC. Only summation is built for transducers; there is
no max mode, recombination, or trace there yet.

### Max-alignment mode

Built for CTC (`--ctc-alignment=max`, `CtcPrefixSearchOptions::alignment`):
a prefix's two sums become maxima, and hypotheses with the same future
recombine as described below (`CtcPrefixSearch::KeepUndominated` in
`ctc-prefix-search.cc`). Its measured effect is in
§ Sum versus max on a full test set. The comparison with `FasterDecoder`
that this section asks for has not been run.

The goal is max-alignment (Viterbi) beam search in the shared decoder,
with only micro-pessimation relative to the current Kaldi `FasterDecoder`
(graehl, 2026-11-12). In this mode, recombination
retains the best compatible path score and its traceback rather than
log-adding alignment mass. Select the accumulation policy outside the hot
loop where practical; sum-mode machinery and unused scorer facilities
must not impose substantial work on the Viterbi path.

The recombination key is model- and mode-specific. A graph-only Viterbi
search can merge histories with equivalent future behavior at the same
graph state; retaining every distinct output prefix would needlessly enlarge
that search. External history-dependent scorers can require a richer key.
Summed output-prefix search has its own identity requirements. Changing
log-add to max without addressing state identity is not sufficient to match
`FasterDecoder`'s search or cost.

With no scorer, max-alignment CTC search reduces to the existing greedy
decoder: the best alignment is the per-frame argmax. The mode therefore
matters only with scorers.

In both modes a hypothesis is a node of the search's prefix trie, and the
node is its identity: alignments that emit the same tokens reach the same
node. Scorer states are stored in the node's own record, so `Advance` runs
once per trie edge. Max-alignment mode additionally merges two nodes when every scorer's
states compare equal and their CTC tails match (last token, and blank or
nonblank ending). Equal states mean equivalent futures, so the
higher-scoring node survives whole, with its score, backpointer, and
per-scorer subtotals. N-best entries past the first are then exact only if
the merged alternatives are also retained.

This is ordinary n-gram recombination, and it must cost what it usually
costs. A scorer's state is a fixed number of bytes, `ScorerInfo::state_bytes`,
in a buffer the decoder owns; the search compares and hashes those bytes
itself, together with the CTC tail, with no call into the scorer. An n-gram
scorer's bytes are its context, written into the caller's buffer as KenLM's
own scoring call does. KenLM's state already shrinks to the longest context
that matched, so histories that back off to the same context merge. The
scorer needs no interning table and allocates nothing per state. A
recurrent scorer's bytes hold a reference to backend-owned state that
depends on the whole history. `ScorerInfo::recombines` says that equal
bytes imply equivalent futures; when one scorer in the set cannot promise
that, recombination is by prefix alone and the search skips the table.
The `FasterDecoder` comparison below applies to n-gram-only scorer sets.

A hypothesis is dropped only when another with the same future is at least
as good both ending in blank and ending in its last token, since either
ending can be the one a continuation needs. The search looks for this
while it fills the beam: it takes candidates best first and passes over
each one that an already kept hypothesis dominates, so only candidates
that reach the front of the ranking are hashed or compared, and the place
of a dropped hypothesis goes to the next candidate. The beam then holds
`max_active` different futures where sum mode's can hold several
hypotheses that an n-gram scorer can no longer tell apart.

Implement CTC blank/repeat/collapse semantics directly in the model policy
for both max-alignment and sum-over-alignments modes. Do not require or
construct an H FST as the CTC mechanism. Optional future graph support is
separate. In max-alignment mode, pluggable direct scorer queries are the
principal added capability; n-gram scoring need not be encoded as traversal
through a precomposed LM graph.

Compare equivalent CTC search workloads against `FasterDecoder` using its
graph representation as the baseline, with matched acoustic inputs, scoring
semantics, pruning controls, and result requests; the new decoder does not
consume that graph. Check recognition/score equivalence with ties handled
explicitly, and measure expansion, pruning, allocation, traceback, memory,
and end-to-end cost. Disable extra functionality in that baseline
comparison and report its enabled cost separately. The existing Kaldi path
stays until this comparison has been run.

## Scorer interface

A scorer adds a log score to a hypothesis as its output grows: a language
model, or anything with the same contract. The interface is
`sherpa-onnx/csrc/scorer.h`; this section says what it commits to.

- **Units.** A scorer consumes recognizer tokens or characters
  (`ScorerUnit`). For characters the search gives it the text the
  recognizer would output for the tokens, as Unicode scalars, with one
  boundary character between words and none before the first; the scorer
  maps the boundary to its model's own unit (`CommonScorerOptions::space`).
  Token text, word starts, and unemittable symbols come from the
  recognizer's token table (`MakeCtcSearchVocabulary`).
- **State.** A scorer's state for one hypothesis is a fixed number of
  bytes, 8-byte aligned, inside the search's record for that hypothesis.
  `ScorerInfo` declares the size and optional `copy_state` and
  `destroy_state` function pointers; null means `memcpy` and nothing to
  release. An n-gram scorer's bytes are its context (KenLM's own state,
  unused bytes zeroed). A scorer whose real state is large or
  reference-counted keeps a pointer in the bytes: the recurrent scorer's
  bytes refer to a reference-counted node per utterance. The search
  destroys every state it started, copied, or received, exactly once.
- **Per-utterance data.** Each scorer creates its own object per utterance
  (`Scorer::NewUtterance`) and receives it with every state
  (`ScorerStateRef`), so a batched call needs no lookup and no state stores
  its utterance. Distinct utterances share model parameters and nothing
  mutable.
- **Calls.** `Start` writes the initial state. `Advance` is batched: one
  unit after each of many parent states, possibly from different
  utterances, writing each child state and its raw natural-log score into
  caller arrays. `Finish` scores the end of input without invalidating
  states. A token of several units is advanced one unit per call, so a
  neural scorer sees each step as one batch across the beam. One
  coordinator calls a scorer serially.
- **Rejection.** `Advance` may reject a continuation and `Finish` a
  hypothesis; the search drops them. The scorers built so far never do.
  A rejection holds even at zero weight; leave a scorer out of the
  configuration to disable it. Model-loading and configuration errors are
  exceptions.
- **Weights.** Scores are raw. The search applies each utterance's weights
  when ranking and keeps raw per-scorer subtotals in its results.
- **Recombination.** `ScorerInfo::recombines` says that equal state bytes
  mean equivalent futures. The search compares and hashes the bytes itself;
  see § Max-alignment mode.
- **Creation.** `ScorerOptions::type` names a registered `ScorerFactory`
  (`RegisterScorerFactory`); `CreateScorer` registers the built-in types on
  first use. Options every type understands are typed
  (`CommonScorerOptions`: unit, model, space, threads, provider); options
  only one type understands travel as JSON text in `extra`, which that
  factory validates.

### Scorer types

- **`kenlm`** (`kenlm-scorer.h`): a character n-gram model in ARPA or
  KenLM binary form. Scores are KenLM's log10 times ln 10; an unknown
  character scores as `<unk>`; the end of input scores `</s>`.
- **`onnx-rnn`** (`recurrent-lm-scorer.h`): a character LSTM or GRU
  exported as one step. The first model input is the previous unit IDs,
  `[batch, 1]`; every further input is a recurrent state tensor
  `[layers, batch, hidden]`; the outputs are next-unit log probabilities
  then the new states. An LSTM has two state tensors and a GRU one, and the
  scorer carries however many the model declares. A node is computed only
  when something is scored after it, batched with the other nodes that
  call needs. `extra` names the unit list file (`vocab`,
  `vocab-encoding`) and the boundary and unknown unit IDs.

### Sketched, not built

The companion keeps the full original sketch. From it, still intended:

- `Next`: a scorer's whole next-unit distribution, with a published
  character-to-index table (`CharacterIndex` exists; the scorers use it
  privately), so the search can use a character LM's first-character score
  before finishing a token's spelling. This is the early partial-score
  beam's input.
- The recognizer's token table passed to scorer factories, for token-unit
  language models and for a scorer vocabulary different from the
  recognizer's (`OurToken`).
- A composite scorer that owns several scorers behind one state.
- A scorer shared by several coordinators or threads.

## Proposed adjustable beam for partially scored expansions

Provide one mechanism for setting or sweeping an early partial-score beam
(graehl, 2026-11-12). A character LM's cached next-unit distribution can
contribute the first character's score before completing every candidate
ASR token's spelling. Further character steps proceed in batches, sharing
common character prefixes. The early beam controls which partially scored
extensions receive that additional work; it is separate from the final
CTC output-prefix beam.

Expose the early threshold as a search-policy input rather than burying it
in a scorer. Fixed settings, an off/exhaustive setting, and a dynamic
controller should all use the same expansion/pruning code. Make the effective
threshold observable so fixed sweeps and adaptive runs can be compared.
The choice of score-gap versus count control, update cadence, and any target
such as a percentage of expansions saved remains open; define the target's
denominator before implementing a controller.

Percentage targets have two intended uses. They can parameterize a convenient
sweep to find an empirically safe-enough pruning threshold, without claiming
a mathematical bound. Alternatively, a runtime controller can vary retained
work to meet a time budget while minimizing quality loss. In the latter case,
use measured time and remaining budget as feedback: an expansion percentage
alone does not enforce a hard deadline, especially with changing cache hits,
batch efficiency, and device contention. Actual hard-budget enforcement needs
a defined stopping/fallback rule as well as the pruning control.

Partial-score pruning can trade recognition quality for reduced scorer work.
Keep it distinct from rejection using a proved upper bound, which preserves
the specified search result. A bound must include remaining scorer and bonus
contributions and account for CTC contributions merging into the same output
prefix. Nonnegative weights on log-probabilities make unfinished LM scores
nonincreasing, but that alone does not justify pruning a merged prefix from
one alignment's partial score.

Instrument candidates considered and rejected, actual character-scoring work
after cache sharing, batch sizes, and time. Compare transcript changes and
error rates against exhaustive token expansion with the same final beam.
Dynamic target policies must report achieved savings rather than imply that
a requested percentage is guaranteed. Building the measurement path and the
bounded expansion mechanism is largely the same work; do not make a separate
measurement-only implementation a prerequisite. Choose defaults and
controller policy from measured tradeoffs.

## Execution placement and batching

Choose recognizer and scorer execution independently. An intended phone
configuration to evaluate is a GPU recognizer with CPU scorers and CPU beam
search. Each scorer has its own provider,
device where supported, and thread settings, without inheriting the
recognizer's placement implicitly. Unsupported combinations must be reported
at configuration time, and effective placement must be observable. CPU-only
backends such as the initial KenLM implementation need not pretend to support
accelerators.

Existing precedent: `OnlineLMConfig` exposes `lm_provider` and
`lm_num_threads` independently of recognizer model configuration.
`ProviderConfig` supplies provider/device and CUDA/TensorRT settings. Reuse
those facilities rather than inventing a separate device vocabulary; details
of mapping applicable fields into each scorer backend remain implementation
work.

The intended calling discipline is one batched scorer call for all its ready
items across the physical batch at a decoding step. Each state carries its
utterance context; differing utterance lengths and weights do not require
separate model calls. Character expansions proceed in dependency waves,
batching all ready next-unit work before advancing to the next wave. Avoid
per-hypothesis neural calls when the work can be grouped.

Beam search initially runs on CPU. Changing the model provider does not move
prefix merging, bookkeeping, or pruning to the GPU. The current `FloatView`
score interface makes that choice explicit; a GPU-resident search would need
additional implementation and potentially a device-resident score interface.
Backend-owned neural states can nevertheless remain on their accelerator.
Keep model execution separate from the search so later implementations can
reuse model and scorer contracts where appropriate. Search on the same GPU
as the models is deferred; see
[the sketches companion](beam-search-and-scorer-fusion.sketches.md#same-gpu-asr-scorer-and-beam-search).

Compare placements on target hardware before choosing defaults. Measure
single-stream latency and sustained batched throughput, scorer batch sizes,
transfer/synchronization costs, and CPU/GPU idle time. On phones, include
sustained performance and whole-device power under thermal constraints;
cold-run speed alone does not decide placement.

### Inflight batching

Decoding admits new utterances while others remain active and finishes or
cancels each without draining the rest; each utterance's search state,
scorer data, weights, and input progress are separate, and ready scorer
work is batched across utterances. `CtcInflightDecoder`
(§ Decoder lifecycle) does this for CTC. The recognizers do a narrower
thing: the utterances or streams handed to one decode call share each
scorer call (`DecodeFramesTogether` in `ctc-prefix-search.h`). The
transducer decoders run one utterance at a time for now.

For offline CTC models the model run and the search are separate calls.
`OfflineRecognizer::EncodeStreams` runs the model and leaves each stream's
log probabilities on the stream; `DecodeStreams` searches streams that
have them and runs the model itself for the rest. The two calls can take
different sets of streams, so a caller can run the model for an utterance
when its recording ends and search it later together with others. Other
model families ignore `EncodeStreams`. On a batch-capable CTC model and on
the omnilingual model, which runs one utterance at a time, results were
identical to `DecodeStreams` alone whenever the model saw the same batches
(2026-11-13, three recordings each, greedy and prefix search).

This covers decoder-side scheduling only. Encoder batching beyond that
seam, capture/segmentation, and scheduling across consecutive utterances
are deferred to [encoder-inclusive inflight scheduling](../gaps/sketches/encoder-decoder-inflight-scheduling.md).

## Model and state representation

Built in `ScoredPrefixTrie` (`scored-prefix-trie.h`), which every model
family's search uses:

- A hypothesis is one record: a fixed header (parent index, last token,
  length, word state, weighted scorer total), then the search's own state
  for the prefix, then one raw subtotal per scorer, then each scorer's
  state bytes (graehl, 2026-11-13; this replaced a reference-counted state
  object per history). Offsets are computed once per utterance from the
  active scorers' declared sizes, each rounded up to 8 bytes.
  `std::max_align_t` is 16 on x86-64 and AArch64 Linux and would pad every
  record for a requirement no scorer has.
- Records sit in fixed-size blocks, so addresses are stable, and refer to
  one another by index. A child is found by (parent index, token). Nothing
  is keyed or hashed by address. An index is reused after its prefix is
  freed; a serial number in the header tells a caller that keeps
  per-hypothesis arrays that this happened.
- A prefix that leaves the beam with no surviving descendant is freed at
  once and its scorer states destroyed.
- New prefixes are scored in waves, one scorer unit per wave, so a token of
  several characters costs a neural scorer one batched step per character
  across the beam, and several utterances' waves can go in one call.
- The CTC search's state is its two alignment sums for the last and the
  current frame, plus the optional trace. The transducer search's state is
  the prefix's probability by the frame it has reached.

Still to try, in the first or second version (graehl, 2026-11-13): keep
each hypothesis's ranking score in a separate contiguous array indexed like
the records, so the pruning scan touches no record. Today the scores are
computed into a per-frame array before pruning, which gives the scan the
same locality without a persistent array; a persistent one would matter
only if scores were reused across frames. Parallel arrays in place of
records, and function-pointer dispatch in place of virtual calls on scorer
models, remain open micro-optimizations to consider if measurements
attribute a significant phone power or temperature cost to layout or
dispatch.

## Decoder lifecycle

`CtcInflightDecoder` (`ctc-inflight-decoder.h`) is the lifecycle for CTC:
`TryAdd` admits an utterance or refuses at capacity, `TryAccept` queues a
chunk of frame log probabilities (copying it) or refuses when the queue
bound would be exceeded, `InputFinished`, `Cancel`, `Step`, `Status`,
`GetResult`, `Release`. Handles are stable and never reused. Each
utterance can carry its own scorer weights and result count. `Step`
advances every utterance that has frames queued by a bounded number of
frames and batches scorer calls across them; an utterance waiting for
input does not hold others back; results match decoding each utterance
alone. A partial result is revisable. Calls are synchronous and made by
one coordinator.

Not built from the original draft (kept in the companion): a model adapter
so one decoder serves other families, asynchronous backends and
`WaitForProgress`, per-device execution options, and timestamps or traces
requested per utterance.

## Model execution boundary (not built)

Nothing in this section is implemented; calls to models and scorers are
synchronous and made in the order the searches need them.

The model-specific loop owns legal expansion and the points at which
successors are complete enough to merge, rank, and prune. The shared
scheduler must not prune a partially completed physical batch as if it were
the full logical candidate set. Model policies expose ready operations with
utterance/hypothesis handles and consume their completed results. They own
CTC collapse, RNN-T label/time advancement, TDT duration/frontier policy,
transformer EOS/cache behavior, and any model-specific safety limits.

Use coarse batches at the executor boundary. An illustrative internal
contract is:

```cpp
struct BatchPlan {
  DeviceId device;
  size_t items;
  size_t workspace_bytes;       // Peak temporary reservation for this call.
  size_t retained_output_bytes; // Includes new states/results surviving it.
  // Backend-owned packing description and dependency handles; no host
  // materialization of recurrent state merely to build this description.
};

class BatchExecutor {
 public:
  BatchPlan Plan(Span<const ReadyOperation> ready, const BatchLimits& limits);
  PendingBatch Submit(BatchPlan plan, DeviceReservation reservation);
  // PendingBatch owns inputs/reservation until completion; exposes readiness
  // and completed outputs. Synchronous implementations complete in Submit.
};
```

The executor schedules calls; `Scorer::Next` and `Scorer::Advance` remain
the scoring interface. A backend adapter can invoke the existing synchronous scorer calls; a later
async backend can provide completion events without changing utterance
lifecycle. Executors declare which operations can batch together, supported
concurrency, and their capacity estimates. Model/scorer state remains
backend-owned on its device; gather/pack there and retain shared parents
across branches. Keep family-specific scheduling outside a universal
per-hypothesis virtual call path where that would cost performance.

### Alternating and overlapping work on the same device

Default to `Alternate`: one active post-encoder ASR or scorer execution per
device, with compatible requests batched together. Alternate by readiness
and dependencies rather than strict ASR/scorer turn-taking; run whichever
class can make useful progress, with bounded fairness across utterances.
CTC normally contributes no hypothesis-dependent ASR calls here, so its
scorers can dominate device scheduling. Separate devices can progress
independently under their own limits.

`AllowOverlap` permits concurrent ASR/scorer calls when the backend supports
it and a shared per-device admission budget can reserve both; whether the
kernels then run concurrently or throughput improves is the backend's to
show. Dependent operations
still wait for their input states/scores; independent hypotheses or
utterances can supply useful overlapping work. CPU search and control
remain separate from GPU execution. This does not implement GPU beam search
or change the CPU-readable `FloatView` contract.

The motivation for alternation is potentially substantial temporary storage
per call, particularly when expansion generates many more successors than
the beam retains. Significant ASR and scorer state remains live across the
time-unrolled dependency graph; alternation does not eliminate that width.
It may nevertheless reduce simultaneously live activations/workspaces and
improve use of the GPU cache hierarchy by reducing interference between
working sets. Both benefits remain measurement hypotheses. Include cache
behavior where observable alongside peak memory, throughput, latency, and
power when comparing schedules.

Independent scorers can execute in parallel when device/backend capacity
permits, or in series under the same resource policy. Serial execution can
also enable early pruning before invoking later scorers. Consider such an
order only when an earlier scorer's weight and influence on ranking can
actually eliminate enough candidates to reduce subsequent model work.
Account for lost batching efficiency and added latency; merely evaluating
a cheap scorer first is not sufficient justification. Expect this pruning
order to be fairly unlikely to be chosen in practice without measured
benefit. Keep execution order separate from the configured scorer indices
used for weights, states, and diagnostics.

Serial early pruning must use a valid bound on remaining contributions to
preserve the specified search result, or be an explicitly configured and
evaluated approximation. Partial weighted scores alone are not a safe
cross-candidate cutoff, especially with negative weights, bonuses, or
alignment mass still to merge. Serial execution without early pruning must
compute the same combined scores as parallel execution, subject to the
backend's numerical behavior.

Budget device memory across resident weights, live encoder/scorer/model
states, retained score/result buffers, allocator-held storage, and the peak
workspace/output reservations of admitted calls. Count shared allocations
once. Reserve before submitting, including allocations for successor states
that outlive the call. On completion, transfer surviving allocations to
live-storage accounting instead of releasing their reservation blindly.
Release input/workspace references only when completion guarantees the
device no longer uses them. Borrowed caller memory must still count toward
device occupancy even when the decoder did not allocate it.

When overlap would exceed capacity, wait for an active call to finish and
run the next batch alone, or reduce its physical batch size. This can happen
well below exhausting device memory if the configured headroom requires it.
Do not trim the logical beam to fit a transient workspace limit. A batch
split must retain the same logical merge/prune boundary and candidate set.
If even one operation cannot fit, report a capacity error; do not endlessly
retry or silently offload/change precision. If retained state alone exceeds
the budget, stop admission or report the unsatisfied configured capacity.

Alternating calls does not unload model weights or guarantee that separate
ORT allocator arenas return cached workspace to a shared pool. Backend
capacity plans must include that retained allocation, and sharing/reuse must
be supported explicitly. Weight swapping, automatic state offload, and
recomputing evicted states are outside the initial plan. Use conservative
backend limits and observable high-water marks where exact planning is
unavailable; an estimate is not a guarantee against external allocations
or provider-internal workspace changes.

Memory occupancy and compute occupancy are different. The initial policy
uses explicit memory and in-flight-call limits, not an assumed portable
measurement of GPU multiprocessor occupancy. An overlap-capable backend may
later use measured load to tune batch size or concurrency. Compare alternate
and overlap modes at the same logical search settings, including sustained
phone power/temperature, server throughput, latency, peak memory, and
numerical/output differences. Keep overlap disabled until the backend's
dependency and lifetime handling is validated; synchronous execution is a
complete first implementation of the lifecycle.

## Options

`--decoding-method=prefix_beam_search` selects the search for offline CTC,
streaming CTC, offline stateless-decoder transducer, and offline NeMo
transducer models. The options are fields of `FusionConfig`
(`fusion-config.h`), a member of the recognizer configs, plus
`--max-active-paths`:

| Option | Default | Meaning |
|---|---:|---|
| `--max-active-paths` | 4 | Hypothesis beam: output prefixes kept per pruning |
| `--score-beam` | 12 | Keep a prefix only within this score of the best; 0 for no limit |
| `--token-candidates` | 8 | Most probable tokens per frame (CTC) or per hypothesis (transducers) that may extend a prefix |
| `--scorer.N.<option>`, `--scorer.<option>` | | Scorers: `type`, `model`, `unit`, `space`, `weight`, `num-threads`, provider options, `extra` |
| `--fusion-config` | | JSON file of any of these options |
| `--length-bonus`, `--length-bonus-unit` | 0, `token` | Added to a prefix's score per token or per word |
| `--ctc-alignment` | `sum` | CTC only: `sum` or `max` (§ Search semantics) |

Scorer options and the JSON form are described in
[option collections](json-option-collections.md). Giving a scorer with
another decoding method is an error. The transducer searches also have a
limit on expansions within one frame (10) that is not an option yet.

Not settled: defaults per model family (the 4 and 12 above were chosen on
CTC; see § Score-gap beam sweep), a shared home for beam options across
decoders ([gap](../gaps/beam-search-options-and-tuning.md)), per-family
option structs, and execution options for device scheduling. The original
table of proposed option structs is in the companion.

## Conceptual search-policy tuning support

Make policy choices inspectable and tunable without rewriting the decoder.
Use the same typed options and execution paths for fixed settings, offline
sweeps, and any later runtime controller. A tuning layer proposes validated
settings; it does not implement a second search algorithm. Begin with fixed
configurations and reproducible sweeps; automatic policy selection and
adaptation build on them later.

Separate three kinds of choice:

- Scoring objective: scorer weights, bonuses, and sum versus max alignment.
  These define what the search ranks, rather than merely its thoroughness.
- Search approximation: beam width/gap, candidate pruning, partial-score
  pruning, auxiliary alignment retention, and model-specific frontier policy.
  These trade work for which candidates or alignment terms are considered.
- Execution: physical batch limits, scheduling quantum, and alternate versus
  overlap. These should preserve the configured logical search, apart from
  documented numerical effects; a capacity limit must not secretly prune it.

Permit model/backend-specific policy choices alongside common numeric
controls. A policy advertises applicable model families, valid parameter
ranges, and the points at which changes can safely take effect. Start by
freezing settings per utterance. A future controller may adjust explicitly
supported controls at complete merge/prune or scheduling boundaries, never
halfway through evaluating one logical candidate set. Widening a beam after
pruning does not recover discarded history; replay would require the
separately deferred recovery machinery. Record changes and their effective
boundaries so adaptive behavior can be inspected and reproduced.

Expose low-overhead counters for generated, merged, scored, pruned, retained,
and auxiliary hypotheses; pruning reasons; scorer calls/items/cache reuse;
physical batch sizes; queue delay; and memory high-water marks. Enable more
expensive timing, traces, device-transfer and cache measurements explicitly.
Keep each observation associated with the effective policy configuration.
Controllers consume these observations and declared limits; the core search
does not depend on a particular optimizer or monitoring framework.

Provide optional per-hypothesis survival-requirement traces. At each logical
pruning boundary, record the smallest occupancy/count limit and the score-gap
(probability-beam) threshold that would retain that hypothesis under the
policy's ordering and tie rule. Carry the maximum requirement encountered
over its ancestry/time steps, separately for every kind of limit: main
hypothesis retention, acoustic candidates, partial scorer stages, auxiliary
alignment retention, and applicable model-specific frontiers. Preserve the
stage/time that set each maximum and the effective scoring configuration.
This reports how wide each limit needed to be for a particular output/path,
not merely how many slots happened to be occupied during the run.

Compute requirements before applying the relevant cutoff, after the merges
required by that policy. Distinguish path ancestry from summed-prefix
support: one retained alignment's trace does not establish the budget needed
to preserve every alignment contribution. For merged histories, define the
trace aggregation explicitly rather than inventing a single ancestor.
Use these maxima as practical estimates of the limits needed to survive.
The design expectation is that changes in competitors induced by tuning
the limits will usually perturb those requirements modestly, making the
traces useful guides for choosing the next settings. This remains to be
checked for the implemented policies, particularly with summed alignment
mass. The estimates are not guarantees that setting every limit to its
recorded minimum reproduces the result. A pruned hypothesis has only a
trace up to pruning; wider reference runs or replay are needed to learn its
later requirements. Attach requested traces to final/N-best results and keep
their collection optional to avoid imposing tracing cost on normal decoding.

Select tracing outside the hot loop, using a specialized no-trace path or
an equivalent mechanism that removes instrumentation work. Non-tracing mode
must not allocate trace records, propagate ancestry maxima, compute ranks
solely for diagnostics, retain otherwise-dead history, or copy score details
solely to support tuning. Ordinary pruning still performs the comparisons
it needs. Cheap operational counters are separately selectable; a disabled
trace sink alone is insufficient if its arguments still require work.

Include scorer weights, length bonuses, and other declared scoring-objective
parameters in tuning support. Retain raw acoustic/scorer subtotals and
explicit bonus-event counts when requested so fixed candidate sets can be
rescored cheaply. Record each bonus's unit and event, and apply parameter
constraints consistently with pruning-bound assumptions. Rescoring a saved
N-best list explores only those candidates: new weights or bonuses can
change prefix survival and scorer work. Validate promising settings by
rerunning search rather than treating fixed-list reranking as equivalent.
Keep weights fixed within each utterance initially; sweeps can reuse loaded
models across utterances with different weights without changing scorer state
semantics.

Tune against a declared objective: for example minimum sustained latency
under a quality/memory constraint, maximum server throughput under a tail-
latency constraint, or acceptable phone power/temperature at a quality target.
Use held-out development audio and an explicitly named error metric, retain
a fixed baseline, and evaluate the selected settings on separate data.
Compare quality/cost tradeoffs, not merely a percentage of expansions saved.
Measure actual sustained hardware behavior; smaller batches or fewer items
do not guarantee lower time or power. Include failure rate where hard
constraints can reject all paths.

Save resolved settings with model/scorer identities, precision, provider,
device, workload/batching conditions, and measurement definitions. Recheck
when those change. In particular, tune alignment-retention budget separately
from the main hypothesis beam, and test serial scorer pruning only when it
can reduce subsequent work. Start with targeted sweeps.

### Reference and oracle exploration modes

Provide optional hooks/options for reference-aware diagnostics and oracle
experiments. Reuse production scoring and policy code; do not maintain an
independent diagnostic decoder with subtly different semantics. Keep these
modes disabled without their scoring, storage, or tracing costs in normal
inference. Reference evaluation and model-specific constrained scoring may
be separate adapters driven by the decoder's events and retained input.

Distinguish the following modes explicitly:

- Honest search: supply a reference for observation/evaluation only. It
  must not change expansion, ranking, pruning, stopping, or the selected
  result. Report reference-prefix survival where observable and score the
  reference separately. Include transcript distance under a named metric.
- Candidate-set oracle: report the reference-nearest output among retained
  candidates, alongside the highest-model-score output. State the actual
  candidate set or N-best/lattice restriction; it is not an oracle over
  outputs the run never explored. Model-score and reference-distance
  selection are different operations.
- Reference-constrained or reference-preserving search: explicitly allow
  constraints or protected candidates to explore reference score and
  survival requirements. Record where protection overrides ordinary
  pruning and the required count/score-gap thresholds. Label these as
  interventions, not honest-search quality or performance results.

For a reachable reference, report its acoustic, scorer, and bonus components
and the score margin `S(reference) - S(search_1best)`. A positive margin
under comparable scoring means the search returned a lower-scoring output;
a negative margin means the objective favors that output over the reference.
Preserve raw margins and optionally report a declared per-character or
per-token normalization for cross-utterance comparison. Exponentiating a
log-probability margin gives a probability ratio only when those scores are
the corresponding log probabilities. Weighted fusion and bonuses generally
produce unnormalized sequence scores, not calibrated `p(reference | audio)`.

Keep search-retained scores distinct from independently rescored sequence
scores. In CTC sum mode, full alignment summation for the reference compared
with only beam-retained mass for 1-best confounds search loss and objective
preference. Rescore both fixed sequences with the same alignment procedure
and accumulation mode to compare them. Report any constrained-scoring beam
or approximation; finite beams do not establish an exact reference score.
Other model families need their own valid constrained-scoring adapter.

Specify whether a reference is an exact token sequence or text with a named
tokenization policy. Different tokenizations of the same text are not
automatically marginalized. Report unrepresentable reference text, excluded
special tokens, and hard scorer rejection separately from finite margins;
do not silently substitute unknown tokens and call the result the reference
score. Retain or replay encoded input only when the requested diagnostic
needs it, with that memory/work included in the exploration budget.

The initial hook design should expose logical pre-prune candidates,
merge/prune decisions, and completed results sufficiently to attach the
survival and oracle diagnostics. Exact callback/event types and supported
constrained-scoring backends remain implementation choices. Validate that
observational diagnostics leave honest-search results unchanged and that
reference-preserving interventions are visible in their reports.

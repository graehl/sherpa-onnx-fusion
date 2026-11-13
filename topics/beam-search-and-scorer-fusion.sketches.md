# Shared beam search and scorer fusion: sketches

Candidate and deferred designs for
[shared beam search and scorer fusion](beam-search-and-scorer-fusion.md).
None is current guidance or part of the first implementation.

## Same-GPU ASR, scorer, and beam search

A future implementation could run ASR and neural scorers on the same GPU,
and move candidate scoring/pruning or more of beam search there as well.
The character/token mismatch does not prevent this: the cached ASR-token
to first-character score-index mapping could live on the GPU and gather LM
scores for ASR candidates. Combining scores and pruning there could reduce
the results transferred to CPU search. Full GPU search would additionally
need device implementations of the model-specific recurrence, merging,
state management, and scheduling.

Intentionally defer device-side lookup, pruning, and beam-search work.
The topic's lifecycle draft separately allows overlapping ASR/scorer model
calls under resource limits; it does not require GPU-resident search.
Keep `FloatView` as the initial CPU-readable
score contract; do not add device-result variants or GPU lookup APIs now.
`Ort::Value` can own CPU or GPU tensors, so retaining one behind a view does
not itself imply GPU execution. CPU tensors can be viewed without copying;
device-only outputs must become host-accessible before serving this contract.
Choosing GPU providers for both models alone does not move search onto the
GPU or eliminate those transfers. Revisit a device-resident result contract
only with a concrete accelerated search design and measured benefit.

## Recovery by iterative widening

Graehl's intended initial scorers will not fail through hard exclusions.
Iterative widening is therefore optional v2/v3 work, not essential to the
initial decoder. It is a search recovery policy, separate from the scorer
interface: retry with a wider search when no accepting completion survives.

Offline ASR can replay saved model outputs. Streaming can also retry if it
retains the history needed to restart search, at a memory and latency cost.
For this CTC decoder, cached frame log-probabilities suffice; retaining encoder
activations is another option if they can regenerate the decoder inputs.
A retry from the beginning needs the whole utterance's history. Retrying
from a checkpoint needs the subsequent inputs and sufficient saved search
state, but cannot recover alternatives discarded before that checkpoint.
Streaming retries must also respect any externally committed transcript
prefix. Buffering scope, retry schedule, resource limits, and behavior when
retries still fail remain future design decisions.

## Cost estimate for a partly consumed scorer token

Different-token LMs also motivate a future cost-estimate addition to the
scorer interface (graehl, 2026-11-12). A recognizer prefix may end partway
through an LM token, leaving an exact LM charge pending. An estimate for
that unresolved contribution would inform pruning instead of treating it
as free. This means scoring cost, not computation time or failure-retention
advice. The estimate needs the scorer history and pending adapter text/state;
their interface boundary remains to be designed. Keep the provisional
estimate separate from accumulated exact scores, replacing it as the pending
text is resolved rather than charging it again. Its sign/units and whether
it is a heuristic or a proved bound must be explicit. No particular estimator
or signature is selected, and the initial matched-text path need not use it.

## Lexicon-like vocabulary mapping

Future L-like use cases could fit the vocabulary request in `ScorerInfo`:
"use this vocabulary of mine; perform the mapping for me." The decoder's
adapter would translate recognition tokens into the requested scorer units,
with the scorer supplying or identifying the mapping/tokenization contract.
A vocabulary alone does not specify how ambiguous segmentations, deferred
emissions, or lexical weights are handled. An explicit lexicon/transduction
adapter is one possible implementation of that contract.

By default this mapping changes only the units supplied to the scorer.
An explicit future mode could also designate the adapter's emissions as the
reported output. That mode must distinguish recognition history from mapped
output history and define their roles in state identity and accumulation;
multiple scorers must not implicitly compete to rewrite the output. This
records L use cases in interface terms, not a promise to accept a composed
HLG graph and infer its L component. Both adaptation beyond the initial
supported units and output rewriting remain future work.

## Conversation and speaker context for scorers

Speculative note, not a plan: an utterance identifies an audio stream and
time interval, ordinarily associated with a particular speaker when
diarization is external to ASR. That speaker may have been observed before.
Future externally supplied conversation time and speaker lists could reach
scorers through `UtteranceScorerContext`. Keep "utterance" for this
across-scorers context (name still provisional); individual scorer states
remain separate objects. Broader context could include conversation,
microphone, and speaker IDs and time spans, with an assurance of temporally
ordered recognition and shutdown or TTL for retained context. These have
plausible applications but no concrete proposal; metadata, ordering, and
lifecycle APIs are all postponed. The initial interface promises none of
them.

## Coarser recombination than the scoring context

Max-alignment search could merge hypotheses on a coarser key than their
scorer state bytes, such as the last two tokens, while each survivor keeps
scoring with its full state (a 5-gram context, or a recurrent state). The
merge is then approximate: a dropped hypothesis could have outscored the
survivor later. It widens the set of distinct futures per active hypothesis
at a fixed beam, and it is the only way a recurrent scorer's hypotheses
recombine at all.

## Auxiliary alignment retention

Planned for a second or third pass (graehl, 2026-11-13): it finds more
different-time CTC derivations of a string the beam already holds, and its
setting will be varied in experiments. The no-LM evidence in the topic
suggests about 0.1 points of ER2 from alignment summation as a whole.

Budget these two kinds of search work independently. The main hypothesis
beam preserves different output histories through LM surprise: a history
that ranks poorly now may become competitive after later acoustic or LM
evidence. Auxiliary alignment retention preserves more timing alternatives
that contribute probability mass to retained output histories. Increasing
one budget does not substitute for increasing the other.

For CTC, propose keeping otherwise-pruned predecessor prefixes as auxiliary
states so delayed emissions can still contribute to surviving descendants.
These states must continue the acoustic recurrence; retaining traceback
objects alone adds no sum terms. Restrict auxiliary expansion to transitions
needed to support retained descendants, with an explicit promotion rule if
an auxiliary prefix is allowed to rejoin the main hypothesis beam.

Proposed controls, with names and defaults still undecided:

- Auxiliary lifetime in acoustic frames after leaving the main beam, with
  zero disabling extra retention. Define when the age starts and whether
  continued usefulness renews it; cap renewal to keep work bounded.
- A separate cap on auxiliary states and an optional score/mass threshold
  for dropping negligible contributors before their lifetime expires.
- The main hypothesis count/score beam, independently tuned for LM surprise,
  and the topic's early partial-score beam for scorer work.

Auxiliary states carry disjoint alignment contributions into the usual
blank/nonblank log-sums. Each alignment contributes once; reactivation must
not replay already-accounted mass or charge the LM again for the same
output extension. Keep prefix-dependent scorer state shared where valid.
Retention still approximates the full sum, and a local mass threshold is
not a bound on the final omitted probability without further analysis.

Sweep auxiliary lifetime/cap at a fixed main beam, then sweep the main beam
at fixed auxiliary settings. Measure extra recurrence work, live memory,
scorer calls, and recognition quality separately. This permits investing in
alignment summation without automatically paying for more distinct LM
histories. These controls apply to sum-over-alignments mode; retaining
extra sum terms has no purpose in max-alignment mode.

## Retention advice for scorers that reject

Left for a second pass (graehl, 2026-11-13). The scorers planned first
never reject; this serves parity with systems that intersect the search
with a grammar. The types below extend the topic's `ScorerInfo` and `Scorer`.

```cpp
// Failure means infinite cost (negative-infinity log score): no accepted
// completion exists, e.g. an FST path that cannot reach a final state.
// This is hypothesis rejection, not a runtime/API error.
enum class FailureCapability { CannotFail, MayFail, Unknown };

struct RetentionAdvice {
  // Expected number of additional per-hypothesis Advance steps until failure,
  // conditioned on failure eventually occurring. A step consumes one scorer
  // unit, not one batch call. From a live state expectation is >= 1;
  // zero is invalid. -1 means unspecified, not CannotFail.
  float expected_failure_delay = -1;
  size_t min_active = 0;              // Zero means no advice; positive counts advise.
  // Documentary estimate of success when min_active states are preserved.
  // Ignored when min_active == 0; not a controller target or guarantee.
  float expected_success_probability = 0;
};

// Added to ScorerInfo:
//   FailureCapability failure = FailureCapability::Unknown;
//   // Guides retention to avoid search failure; ignored when CannotFail.
//   RetentionAdvice advice;

struct AdviceContext {
  std::optional<float> remaining_units; // Only when meaningfully available.
};

// Added to Scorer. Optional cheap/cached refinement. Nullopt leaves static
// advice in force; present advice is a replacement, with unknown numbers
// kept unknown.
//   virtual std::optional<RetentionAdvice> Estimate(
//       const StateRef& state, const AdviceContext& context) const;
```

Advice fields are heuristic; proved bounds would need separately typed
fields rather than overloading these expectations.

User-directed, 2026-11-12: the scorer API should optionally provide static
and state-dependent estimates to inform a desired “succeed without
backtracking” minimum-active policy. Success means an accepting completion
under the configured constraints, not necessarily the highest-scoring
completion. Whether an empty transcript is acceptable is a separate search
policy; emitting something once does not establish successful completion.

Provide a static scorer-info struct containing an explicit failure capability
and two advice quantities: expected additional `Advance` steps until failure,
conditioned on failure eventually occurring (`-1` means unspecified),
and a plain minimum-active count. Zero is the default sentinel for no advice;
one explicitly advises preserving at least one state. Neither requests an
empty beam. The
associated expected success probability documents the expected effect of
preserving that many states, such as 99%; it is not a requested controller
target or a guarantee. Provide a state-dependent query for corresponding
refinements. Count per-hypothesis transitions consuming one scorer unit,
not physical batch calls or acoustic frames. A live state needs at least one
additional step to fail, so a specified expectation is at least one; zero
is invalid. This estimate concerns failure reached through advancement;
end-of-input rejection is reported separately by `Finish`. Failure capability
remains separate from unspecified delay or zero retention advice.

Represent failure capability explicitly, for example with `CannotFail`,
`MayFail`, and `Unknown`, defaulting to `Unknown`. `CannotFail` promises that
the scorer itself never rejects a continuation or end-of-input completion
within its declared input domain. It does not promise freedom from runtime
errors or acceptance by other scorers or graph constraints.

A `MayFail` scorer with `min_active` zero or one still requires retry
eligibility. Neither an unspecified delay nor that minimum-active value means
`CannotFail`. Retry eligibility is determined separately from numeric advice,
considering all active constraints; an unknown capability must not silently
disable recovery. Known cannot-fail scorers are the common case and need no
failure-prevention retention on their own account.

Distinguish the resulting search cases explicitly:

| Active configuration | Preventive retention | Failure-triggered retry |
|---|---|---|
| All scorers are `CannotFail`, with no other failure-producing constraints | None needed for failure prevention | Disabled; never retry for this purpose |
| Failure is possible; minimum-active advice is zero | No advice; use configured search policy | Eligible |
| Failure is possible; minimum-active advice is positive, including one | Use the advised count and its documented success expectation | Eligible |
| Failure capability or advice is unknown | Documented fallback | Eligible, subject to the configured recovery policy |

When recovery is implemented, resolve its eligibility independently of
retention advice. The cannot-fail configuration need not retain replay
history solely for failure recovery. An unexpected failure in that mode
is a violated assumption or another error to report, not a trigger for
silently widening the beam.

Additional optional information includes:

- Whether the scorer imposes hard exclusions or permits all continuations,
  and whether completion is currently allowed, impossible, or unknown.
- Lower/upper bounds or expectations for additional scorer units before
  completion, or before a deferred constraint can reject a continuation.
- An estimated survival probability over a stated continuation horizon,
  when the scorer has a defensible model for it.

Keep proven bounds distinct from heuristic estimates. Quantities name their
units and event; probabilities or expectations also state their conditioning
assumptions. Scorer characters or tokens are not acoustic frames, and a
delay until more scoring evidence arrives is not a bound on time to failure.
Do not add token rewriting to the scorer interface to obtain these hints.
Any future graph search may contribute analogous graph-specific information.

The search owns the policy that combines these hints with the current beam
and available input horizon to choose an adaptive minimum number of active
hypotheses. Scorers report information rather than prescribing beam width.
Keep score-based pruning, a configured minimum, and a resource maximum
explicit; report when the resource maximum prevents the desired retention.
Without estimates, use a documented fallback rather than treating missing
information as certainty of success or failure.

Different scorers' constraints can conflict, and different hypotheses can
fail together. Individual feasibility does not establish joint feasibility;
do not multiply survival estimates under an unstated independence assumption.
Estimates guide retention but do not guarantee success or justify claiming
that an input has no accepting path. Retaining more hypotheses also cannot
recover extensions already removed by candidate pruning.

The estimate types, query cost/caching contract, combination rule, and mapping
to minimum-active counts remain design work. Prefer cheap or cached queries
that preserve deferred neural computation. Before adopting an automatic
default, compare failure frequency and cost against fixed minimum-active
retention, including misleading or unavailable estimates and correlated
dead ends. For the horizon needed to expose all scorers' deferred constraints,
use the maximum of their delay estimates after conversion to common units.
The retention count needed for a target success probability over that horizon
remains a joint-search estimate.

Recovery when every hypothesis is rejected is sketched under
[Recovery by iterative widening](#recovery-by-iterative-widening).

## Score-gap beam default from model score ranges

The score-gap beam's default is a constant, 12, from one sweep. A default
that holds across models would be derived from how far apart competing
hypotheses' scores sit: the recognizer's score scale, and each scorer's
weight and the range of its scores. That needs each model to publish its
score range, by a metadata convention or as an explicit option, and a rule
that turns those ranges and the weights into a gap (graehl, 2026-11-13).
Until then, anyone who cares tunes `--score-beam`.

## Original design text for parts since built

Kept for the reasoning and for the pieces not built. Where this disagrees
with the topic or the code, they are current and this is not.

### Status lists as of the design

The goal is efficient phone and server ASR inference with LM fusion. Add
sum-over-alignments CTC fusion and improve the non-CTC use cases that already
have fusion support. Success means useful recognition quality at acceptable
latency, throughput, memory, and sustained power/thermal cost on the target
hardware. The interfaces and code-sharing choices below serve that goal.

This topic is a design: none of the decoder, its interfaces, or its CLI
options exists in the code yet. Names, signatures, and numeric defaults are
provisional until the first implementation is reviewed. Implement and prove
the decoder in this fork first; a contribution to Kaldi can follow.

### First implementation

- CTC prefix search that sums alignments, on CPU, with or without scorers
  and without an OpenFst graph.
- Scorers that never reject: an ONNX recurrent LM (LSTM or GRU) and an
  optional KenLM n-gram, in `AsrToken` or `Character` units, several at
  once with independent weights.
- LMs trained on the recognizer's own text convention, so token-to-unit
  expansions are precomputed and no text is normalized at run time.
- Batched scorer calls across utterances and inflight batching, driven by
  one coordinator making synchronous calls.
- Scorer configuration by `--scorer.` options for one or several scorers,
  or from a JSON file; search settings fixed per utterance.
- Pruning by score beam and minimum and maximum active hypotheses.
- Correct sums on tiny enumerable examples and across chunk boundaries
  before any quality comparison with greedy search.

Max-alignment mode, the early partial-score beam, and the tuning traces are
specified below without an assigned pass. Later passes add policies for
transducer, TDT, and autoregressive models, overlapping device calls,
auxiliary alignment retention, retention advice for scorers that reject,
and scorer-specific token units. The
[sketches companion](beam-search-and-scorer-fusion.sketches.md) holds the
designs that are not current guidance.

### Implemented so far

The code is the authority for what exists; the sketches below remain the
design for what does not.

- **Scorer interface and registry**: `sherpa-onnx/csrc/scorer.h`,
  `scorer-options.h`. It differs from the sketch below in four ways. Each
  scorer creates its own per-utterance object (`Scorer::NewUtterance`) and
  receives it with every state, so a batched call needs no lookup from a
  shared context. `Advance` and `Finish` write into caller arrays instead of
  returning vectors. `Next` and `FloatView` are not built, because the
  search batches a token's units through `Advance`. The unit a character
  scorer's model uses for a word boundary is the common option `space`.
  Type names carry a `Scorer` prefix (`ScorerStateRef`, `ScorerExtension`).
- **CTC prefix search**: `sherpa-onnx/csrc/ctc-prefix-search.h`, one object
  per utterance, sum and max alignment. In max mode, when every scorer
  sets `recombines`, a hypothesis is dropped when another with the same
  last token, word state, and scorer state bytes scores at least as well
  both ending in blank and ending in that token; the two endings are
  compared separately because a repeated token extends only the blank
  ending. The best result is unchanged; later results are no longer each
  output's own best alignment.
  A frame runs in phases (`BeginFrame`, `GatherScorerWave` and
  `ApplyScorerWave` per scorer, `EndFrame`) so a caller decoding several
  utterances can put their scorer work in one batch; a token of several
  scorer units takes one wave per unit. Records live in fixed-size blocks
  and are addressed by index; a prefix out of the beam with no surviving
  descendant is reclaimed at once. Candidate tokens per frame are the
  `max_token_candidates` most probable. The length bonus counts tokens or
  words. A timestamp is the frame whose emission of the token carried the
  most probability. Tests compare unpruned search with enumeration of all
  alignments.

- **Scorer type `kenlm`**: `sherpa-onnx/csrc/kenlm-scorer.h`, built only
  with `-DSHERPA_ONNX_ENABLE_KENLM=ON` (default off; KenLM is LGPL and is
  fetched at configure time, query sources only, no Boost). A character
  n-gram model in ARPA or KenLM binary form. Its state is KenLM's own, with
  unused bytes zeroed, so equal contexts are equal bytes.

- **Offline recognizer**: `--decoding-method=prefix_beam_search` for
  offline CTC models, with `--max-active-paths` as the hypothesis beam,
  `--scorer.*` or `--fusion-config` for scorers, and `--length-bonus`,
  `--length-bonus-unit`, `--ctc-alignment`, `--ctc-token-candidates`,
  `--ctc-score-beam`
  (`sherpa-onnx/csrc/fusion-config.h`,
  `offline-ctc-prefix-beam-search-decoder.h`). The utterances of a batch
  share each scorer call. Token text, word starts, and unemittable symbols
  come from the recognizer's token table; byte-BPE tables are refused. A
  character scorer scores the text the recognizer would output for the
  tokens, except that no word boundary is scored before the first word or
  twice in a row. Join markers the recognizer does not remove itself
  (`##`, `@@`) would be scored as written; no such model is supported. The
  options exist in the C++ config and command line only; see
  [the bindings gap](../gaps/fusion-config-bindings.md).

- **Streaming recognizer**: the same method and options for streaming
  CTC models (`online-ctc-prefix-beam-search-decoder.h`). Each stream keeps
  its search across chunks and drops it at an endpoint; streams decoded in
  one call share each scorer call. A stream's result is the best prefix so
  far, which later frames can revise. Once the stream's input is declared
  finished, each further decoded chunk reports the result with the
  scorers' end-of-input scores, so the last one is final; a stream finished
  with nothing left to decode keeps its last result. The trailing-silence count used for endpointing is the number of
  frames since the best prefix's last token was most probably emitted.
- **End-to-end check**: `sherpa-onnx-offline` with a CTC model gives the
  same transcripts with `greedy_search` and `prefix_beam_search` on two
  short test recordings, and accepts a KenLM scorer from `--scorer.*` and
  from `--fusion-config`. The streaming path has unit tests only.

- **Multi-utterance decoder**: `sherpa-onnx/csrc/ctc-inflight-decoder.h`,
  the lifecycle drafted below for CTC: `TryAdd`, `TryAccept` (copies the
  chunk), `InputFinished`, `Cancel`, `Step`, `Status`, `GetResult`,
  `Release`, with stable handles, per-utterance weights and result count,
  an admission limit, and a bound on queued input. `Step` advances every
  utterance that has frames queued and batches scorer calls across them;
  an utterance waiting for input does not hold others back. Calls are
  synchronous, end-of-input scoring is per utterance, and the recognizers
  do not use this class yet: they batch within one `Decode` call.

- **Survival trace**: `trace_survival` adds 40 bytes to each record when
  on and nothing when off, and reports per result the score gap and
  hypothesis-beam rank it needed; a search given those reproduces the
  result (see the sweep section for what that means in each mode). Other traces would be further optional
  byte groups in the record, each with its own switch.

Not built yet: the early
partial-score beam and the other tuning traces.

### Scorer contract and C++ interface sketch (as designed)

## Pluggable scorer contract

The intended contract follows issue #4186:

- A scorer's unit is the symbol it scores one at a time: a character or a
  token (`ScorerUnit` below). `advance(state, unit)` returns successor state
  and the unit's log score. `next(state)` optionally supplies all next-unit
  scores.
- Each scorer declares its unit and applicable mapping; configuration
  supplies its scale. Multiple scorers can participate, including an n-gram together
  with a neural LM. Character scoring expands an emitted token's text
  into LM characters.
- Compute and cache `next` only on first demand for surviving prefixes.
  Defer neural recurrent computation to that point and batch compatible
  states. A prefix revisited on another acoustic frame reuses its cache.
- Use combined acoustic and scorer scores to choose extensions when
  `next` is available. A scorer exposing only `advance` scores the chosen
  acoustic candidates. Candidate pruning remains an explicit approximation.
  For a character LM, a token extension can require several character
  steps; its first-character distribution alone is not the token's score.
  Efficient token-candidate scoring across those steps remains an API detail
  to resolve.
- Fork scorer states with hypotheses and release unused states on pruning.
  Retain separate scorer subtotals and allow returning them in the final
  beam for diagnostics and subsequent weight tuning.
- Reuse the existing ONNX RNN-LM model interface. An optional KenLM backend
  remains behind a default-off CMake option; the search must not require it.

#### C++ interface sketch

This pseudocode extends issue #4186 with explicit state ownership, batched
operations, and factories. Names and container types are provisional;
`Span` means a borrowed range in whatever form the project's C++ standard
allows.

#### Configuration, factory, and model information

```cpp
// AsrToken: recognizer symbol IDs. OurToken: scorer-specific IDs (future).
// Character: Unicode scalar values in the ASR tokens' existing text convention,
// without normalization or case conversion. Other encodings/forms are TBD.
enum class ScorerUnit { AsrToken, Character, OurToken };

// Each has the project's usual Register(ParseOptions *), Validate(), and
// ToString(). CommonScorerOptions registers without a prefix, so the option
// paths are scorer.N.model, scorer.N.unit, and so on.
struct CommonScorerOptions {
  std::string unit = "char";           // "char" or "token"; resolved to ScorerUnit.
  std::string model;
  ProviderConfig provider;             // Existing provider/device configuration.
  int32_t num_threads = 1;             // Independent of recognizer settings.
};

struct ScorerOptions {
  std::string type;                    // Required registered factory name.
  CommonScorerOptions common;
  float weight = 1.0f;                 // Default search weight for this scorer.
  std::string extra = "{}";            // JSON object text for the factory.
};

// Built during scorer setup, immutable once published through ScorerInfo.
struct CharacterIndex {
  // Unicode scalar -> index in the scorer's Next log_scores array.
  // Returns -1 for an unmapped scalar or an out-of-range codepoint.
  int32_t Lookup(char32_t codepoint) const {
    return codepoint < indices_.size() ? indices_[codepoint] : -1;
  }

  // Requires a Unicode scalar and a nonnegative score index; throws otherwise.
  // Extends with -1 entries as needed, then assigns this mapping.
  // Repeating the same mapping is allowed; conflicting assignments throw.
  void Set(char32_t codepoint, int32_t score_index);

 private:
  // Empty if nothing is mapped; otherwise max mapped codepoint + 1 entries.
  // Internal gaps are -1. No trailing padding beyond the last mapped scalar.
  std::vector<int32_t> indices_;
};

struct ScorerInfo {
  ScorerUnit unit;                     // Resolved unit, including defaults.
  // OurToken (future): a token LM builds an index for efficiently tracking
  // ASR tokens -> decoded text -> its own tokens as each ASR token is appended.
  // Incremental mapping state/API TBD; no reverse token-ID-to-string table here.
  // Loaded with the model: actual Unicode scalar -> model vocabulary ID.
  // For Next-capable character scorers, that ID also indexes Next scores.
  CharacterIndex characters;          // Character only; empty otherwise.
  bool supports_next = false;
  // Per-hypothesis state: this many bytes in a buffer the decoder owns.
  size_t state_bytes = 0;
  // Optional. A null copy_state means memcpy; a null destroy_state means
  // nothing to release. A scorer whose real state is variable-length or
  // reference-counted keeps a pointer or pool index in the bytes and sets
  // both. copy_state writes an uninitialized buffer.
  void (*copy_state)(const std::byte* from, std::byte* to) = nullptr;
  void (*destroy_state)(std::byte* state) = nullptr;
  // True when equal bytes imply equivalent futures (an n-gram context).
  // Requires canonical bytes: unused bytes zeroed.
  bool recombines = false;
};

// Immutable ID-indexed recognizer vocabulary plus token-to-text conventions
// and special-token metadata. Exact representation remains to be specified.
class AsrTokenSet;  // Placeholder for access to the recognizer's existing vocabulary.
using AsrTokens = std::shared_ptr<const AsrTokenSet>;

class Scorer;
class ScorerFactory {
 public:
  virtual ~ScorerFactory() = default;
  virtual std::unique_ptr<Scorer> Create(
      const CommonScorerOptions& common,
      std::string_view extra,
      AsrTokens asr_tokens = {}) const = 0;  // Optional cache preparation.
};

class ScorerFactoryRegistry {
 public:
  // All registration is serial and completes before any CreateScorer call.
  // Takes ownership. Empty names, null factories, and duplicate names throw.
  void Register(std::string name, std::unique_ptr<ScorerFactory> factory);

  // Throws for an unknown name; reference valid for the registry's lifetime.
  const ScorerFactory& Lookup(std::string_view name) const;

 private:
  std::unordered_map<std::string, std::unique_ptr<ScorerFactory>> factories_;
};

// One global registry shared by built-in and application factory registrations.
ScorerFactoryRegistry& GetScorerFactoryRegistry();

// Explicit serial startup registration, before any scorer creation.
// Includes only backends enabled in the build; call once during startup.
void RegisterBuiltinScorerFactories();
// Applications call GetScorerFactoryRegistry().Register(name, custom_factory).

// Resolve one factory and construct its model with common/backend options.
std::unique_ptr<Scorer> CreateScorer(
    const ScorerOptions& options,
    AsrTokens asr_tokens = {});  // Uses the global registry.

// Convenience wrapper: CreateScorer for each entry, preserving input order.
std::vector<std::unique_ptr<Scorer>> CreateScorers(
    const std::vector<ScorerOptions>& options,
    AsrTokens asr_tokens = {});
```

A scorer entry's `type` names its registered factory. Fields that only one
factory understands go in `extra`, a JSON object passed to that factory as
text. Each selected factory validates its own options, including unknown
keys. In-process parsing can use
the existing `nlohmann::json`; JSON objects need not cross a future shared
library boundary. A plugin factory could receive the same JSON string and
return a scorer through an adapter; shared-library loading and its ABI come
later.

An optional ASR token set at factory time permits cache preparation; a null
handle skips it. It does not restrict the scorer to that vocabulary. The
same scorer can serve utterances from different ASR models in one batch
without another factory build. Each utterance context supplies its actual vocabulary;
validate compatibility for that vocabulary and use lazy caches or uncached
adaptation when no prepared entry exists. No separate bound-scorer class is
required.

`AsrTokenSet` names access to the recognizer's existing output-token IDs and
vocabulary, not a new vocabulary defined by this decoder. CTC and transducer
recognizers already use `SymbolTable`. For character LMs, preview the
decoded text of each output token by ID, including the recognizer's handling
of word-boundary markers and spaces; raw token-file spellings are not the
character sequence to score. Never use `sym2id()` for this: it reverses the
needed direction and loses duplicate spellings at distinct IDs.

`SymbolTable::operator[](id)` currently returns a transformed string copy;
some representations also require sequence-level decoding. Efficient access
and caching remain TBD. We may enhance `SymbolTable` in concert to expose
decoded token views or reusable decoded storage without redundant copying.
Do not bypass required decoding merely to obtain a zero-copy raw string.
The initial complete-Unicode-token path can precompute expansions; general
cross-token byte decoding remains deferred as described below.

For the `Next` optimization, require efficient token-ID-to-first-output-char
access, preferably without returning a string copy. This may be supplied by
`SymbolTable` itself or by a compiled interpretation using configuration and
existing ASR-model metadata queries. Return a Unicode scalar, not the first
UTF-8 byte, and honor decoded spaces and model-specific special tokens.
Distinguish a known first character from no emitted character or an unresolved
context-dependent result; never guess from the raw token spelling. Only a
known first character enables this particular early-score lookup.

On the initial supported path, precompute the first character and its
scorer-specific `Next` index for each ASR token. Candidate expansion then
uses indexed access without string allocation/copying or repeated Unicode
decoding. Full character expansions remain available for subsequent scoring.
The accessor's exact API and placement remain TBD; do not require a duplicate
string vocabulary merely to supply this operation.

The decoder owns reusable token-to-character mappings and exposes them
through the context. For v1, shared ownership keeps the ASR resources alive
until all scorers, decoders, and cached mappings using them release ownership;
weak-cache expiry and explicit unload notifications are unnecessary initially.
Cache identity includes token IDs,
text interpretation, and special-token metadata, not just vocabulary size.
Immutable tables can be shared across utterances and scorers; synchronize
lazy insertion and retain backing storage while entries reference it.
History-dependent LM scores remain separate from these static mappings.

Use one global scorer factory registry, accessed through
`GetScorerFactoryRegistry()`. All registrations, built-in and application,
occur serially and finish before the first `CreateScorer` or `CreateScorers`
call. No registration, replacement, or removal is allowed after creation
begins, even between otherwise nonconcurrent creation calls. Registration
never silently replaces an existing name. Startup coordination belongs to
the application; merely making singleton initialization thread-safe does
not make concurrent registration valid.

After that registration phase the registry is read-only, and scorer creation
may be concurrent. Factory `Create` implementations must support concurrent
calls or synchronize their own mutable resources. Keep the global registry
alive through all creation calls; created scorers do not borrow the factory
or registry. Explicit built-in startup registration avoids dependence on
cross-translation-unit static initialization order. A future plugin factory
must transfer shared-library lifetime ownership to its created objects as
needed; late plugin registration is outside this contract.

Common model options are unit (default `Character`) and a
model path (empty means no common path supplied). A backend that requires
a path validates its presence. Loading the model normally loads its native
vocabulary too, whether embedded or supplied in the model's own sidecar files.
Do not require an independently translated vocabulary format or a common
`symbols` option. Backend-specific configuration can locate native sidecars
or supply interpretation information genuinely absent from the package.
Factory selection belongs to the registry,
not the backend options. Provider/device selection and thread count are
independent common settings for each scorer; reuse the existing provider
configuration machinery where applicable. Neural tensor names,
KenLM-specific settings, and a plugin's alternative config filename belong
in its JSON until a shared contract warrants promoting them. Factories must
copy or parse borrowed strings before returning; options need not outlive
the constructed scorer.

Weight defaults to 1 in the scorer entry and is not passed to its factory.
`DecoderOptions` carries the list of scorer entries and a parallel list of
weights, taken from the entries unless set directly. Those are the weights
for every utterance unless `UtteranceOptions::scorer_weights` overrides
them, which a server handling differently tuned requests may do. A model
can score a batch of utterances with different weights;
states and cached distributions contain only raw scores. The search applies
its weights when ranking, preserving raw subtotals for diagnostics and retuning.

Use ordinary fields where omitted and explicitly supplied default values
have the same meaning. Introduce default-provenance tracking only for a
specific warning or fallback policy that needs the distinction, rather than
wrapping every option. Factories return resolved model info so the search
does not guess the effective symbol table or unit.

The initial ONNX recurrent scorer should support both LSTM and GRU. Reuse
the existing LSTM export convention as a default, while allowing backend
configuration/model metadata to specify tensor names and state layouts.
GRU has one recurrent hidden state; it must not need a dummy LSTM cell state.
Keep this adaptation in the backend, independent of the generic scorer API.

#### Vocabulary exposed by Scorer

`const ScorerInfo& Scorer::Info() const` exposes the resolved vocabulary
mapping after model loading. The scorer owns it; the returned reference and
mapping remain valid and immutable for the scorer's lifetime, including
concurrent calls. The backend interprets native vocabulary spellings and
trainer conventions when building the mapping, so the decoder sees actual
text. For example, a model using `▁` to encode space maps U+0020 to that
model entry. Do not infer this substitution merely from seeing `▁`; it must
follow the model's convention. BOS/EOS and unknown-token behavior stay with
the backend's `Start`, `Finish`, and unsupported-character handling, not
fake Unicode characters in the map. No reverse string table is required.

`AsrToken` uses the recognizer's IDs and vocabulary through the utterance
context. Neither its options nor its `ScorerInfo` duplicates that table.
`OurToken` distinguishes a scorer's different vocabulary. A token LM will
build an index that efficiently tracks ASR tokens -> decoded text -> its own
tokens as each ASR token is appended, retaining incremental mapping state
where token boundaries differ; this may require word-boundary awareness.
Build it from the vocabulary loaded with the model. Its representation and
API remain TBD; `ScorerInfo` does not expose a reverse token-ID-to-string
array.

For `Character`, the interface uses Unicode
scalar values directly, retaining the ASR tokens' existing text convention.
There is no implied NFC/NFD conversion. A character backend publishes its
character-to-model-ID mapping as `ScorerInfo::characters`, deriving it from
the model's native vocabulary. For `Next`, model IDs index the returned
score array. This mapping is also exposed when `Next` is unsupported;
the backend can use it for `Advance`, which still takes Unicode
scalars. The decoder caches the mapped indices for ASR-token expansions;
in particular, comparing `Next` scores against the first character of each
ASR choice needs only the cached index. A missing mapping returns -1 and
must be handled before indexing scores, according to the backend's declared
unsupported-character behavior. Every published nonnegative index must be
within that scorer's `Next` score array.

Initially `CharacterIndex` uses direct indexing with internal -1 gaps and
exactly `max_mapped_codepoint + 1` entries, or an empty vector for no mappings.
Paged, hashed, or sorted representations remain alternatives to consider;
the lookup API hides that choice. No reverse index-to-character mapping is
required. Decide whether to retain one when adding debug printing.
Other character encodings and normalization forms are TBD.
The initial supported modes are `AsrToken` and `Character`.
Scorer loaders use their model package's existing vocabulary representation;
there is no requirement to convert it to Sherpa's `tokens.txt` or another
derived sidecar. The ASR recognizer's own vocabulary loading remains separate.

Coordination can live inside one composite `Scorer`: its factory parcels out
subconfigurations and calls subfactories, and the object owns its subscorers,
coordinates their evaluation, and exposes one state/score contract to the
search. The ordinary C++ virtual-object interface supports this without
giving every scorer indexed access to its peers' objects or state.

Defer direct peer access as a possible small optimization, in the same spirit
as deferring specialized state layouts. No use case has been identified where
it matters. Require a concrete use case and demonstrated benefit before adding
that coupling to the general interface; composition is sufficient initially.

#### State and scoring operations

```cpp
using UnitId = int32_t;  // Unicode scalar for Character; ASR/own symbol ID otherwise.

class UtteranceScorerContext {
 public:
  virtual ~UtteranceScorerContext() = default;
  AsrTokens asr_tokens;                // Actual vocabulary for this utterance.
  // Decoder-supplied utterance storage; allocator/pool access API still open.
  // Backend-specific caches/pools are partitioned by scorer instance.
  // Conversation and speaker context could reach scorers here later; see
  // the sketches companion. The initial interface carries none.
};
using Context = std::shared_ptr<UtteranceScorerContext>;

// A scorer's state for one hypothesis is Info().state_bytes bytes, 8-byte
// aligned, inside the decoder's record for that hypothesis. The decoder copies
// and discards states with ScorerInfo's copy_state and destroy_state when
// they are set, and with memcpy and nothing otherwise.
struct StateRef {
  UtteranceScorerContext* context;    // The state's utterance; outlives the call.
  const std::byte* bytes;
};
struct Extension {
  StateRef parent;
  UnitId unit;
  std::byte* child;                   // Uninitialized; written unless rejected.
};
struct AdvanceResult {
  float log_score;                    // Raw natural-log score.
  bool rejected = false;              // Child left unwritten.
};
// Read-only contiguous CPU floats with shared lifetime ownership.
struct FloatView {
  const float* data = nullptr;
  size_t size = 0;
  std::shared_ptr<const void> owner;

  // Nonempty views require an owner keeping the entire range alive.
  // Subviews share ownership; no float copy. Index must be < size.
  float operator[](size_t index) const { return data[index]; }
};

struct NextScores {
  // Character: Info().characters.Lookup(codepoint) gives the score index.
  // AsrToken/OurToken: the dense token ID is the score index.
  FloatView log_scores;
};
struct FinalScore {
  bool accepted;
  float log_score;                    // Raw EOS/final contribution.
};

class Scorer {
 public:
  virtual ~Scorer() = default;
  virtual const ScorerInfo& Info() const = 0;
  // Initial context/BOS. Writes the initial state into an uninitialized buffer.
  virtual void Start(UtteranceScorerContext* context, std::byte* state) = 0;

  // Called only when Info().supports_next. Implementations return one result
  // per state, in order. Unsupported scorers inherit the empty default.
  // A character LSTM exposes its next-character distribution in Character
  // units, not scores for whole multi-character ASR tokens. A token-trained
  // LSTM can expose Next in its own matching token units.
  virtual std::vector<NextScores> Next(Span<const StateRef> states) {
    return {};
  }

  // Required; also works for scorers without Next. One result per extension.
  virtual std::vector<AdvanceResult> Advance(
      Span<const Extension> extensions) = 0;

  // Required; called at actual end of input, not at every streaming chunk.
  virtual std::vector<FinalScore> Finish(Span<const StateRef> states) = 0;
};
```

`FloatView` separates the score array from its allocation mechanism. Prefer
caller-allocated output buffers supplied to ONNX Runtime when output shapes
are known: the backend retains the allocation through `owner`, and ORT writes
into it. Do not reuse or overwrite that buffer while any returned view is
live. Alternatively, retain an owning `Ort::Value` behind `owner` and point
directly into its CPU tensor. A GPU result requires a completed transfer to
host-accessible storage before publishing this CPU view; that storage uses
the same lifetime contract. Batched output rows can share one owner.

Keep `FloatView` for now; `std::shared_ptr<Ort::Value>` is a reasonable
alternative with comparable design merits.
`FloatView` accommodates native score storage from arbitrary non-ONNX
scorers without requiring an ORT wrapper or a copy when wrapping their
storage is awkward. The ORT alternative carries tensor shape, type, and
device information directly, and already fits this project's use of ORT
in C++ model interfaces. Different devices would select execution or
transfer paths rather than invalidate that interface. Caller-backed CPU
ORT tensors can also avoid copies, provided their backing owner is retained.
There is no measured performance winner between these interface choices.

The opaque shared owner includes the deallocation action and an atomic
reference count. Copying/destroying views incurs reference-count operations;
float access does not. This is not cost-identical to a move-only buffer with
a callable deleter, which needs no shared count. Exact object/control-block
layout is implementation-dependent. Keep shared ownership initially and
consider cheaper lifetime machinery only under the measured-overhead policy
below. The view does not decide whether values are logits or normalized
scores; `NextScores::log_scores` continues to mean natural-log probabilities.

Operations are batched per scorer instance. Results preserve request order;
`Advance` scores one unit, and its score agrees with `Next`
for the same state and unit when both are supported. `Next`
supplies scores indexed by its vocabulary mapping. The backend must define
unsupported-character behavior, including any native unknown-token fallback;
an unmapped character must never be used as an array index.

The first scorers never reject, but the interface lets a scorer reject so
that grammar-constrained scorers fit later. Rejection returns negative
infinity and no child state. `Finish` reports acceptance and any final score
separately from runtime errors. Model-loading, invalid-config, and execution
errors use exceptions at the owning boundary. A hard rejection remains a
constraint even at zero scorer weight; exclude the scorer from configuration
to disable its constraints. Never multiply a rejected negative-infinity
score by zero. Retention advice for scorers that reject, and recovery when
every hypothesis is rejected, are
[sketched for a later pass](beam-search-and-scorer-fusion.sketches.md#retention-advice-for-scorers-that-reject).

A state's bytes represent an immutable logical history. Forking a
hypothesis copies them; advancing writes a child without changing its
parent. The decoder discards, through `destroy_state` when set, every state
it started, copied, or received from `Advance`, exactly once and before
destroying the scorer. A backend that wraps shared caches or model
resources in its states therefore sees every pruned prefix released and
must not keep a model-global cache that outlives them. A future plugin
adapter keeps its library loaded through destruction of every associated
object. Initially one coordinator
makes all calls to a scorer instance serially, and each call can batch
states from many utterances. Calling one scorer instance from several
threads or several decoders requires an explicit later contract.

Use fully separate reference-counted arenas/pools for each utterance
(graehl, 2026-11-12). The decoder supplies an utterance-owned scorer context
to each scorer's `Start`. Scorers can allocate from its storage or retain
backend-specific pools under that lifetime; each scorer's caches are kept
separate within the context. `Advance`, `Next`, and `Finish` receive the
context with each state. Mixed-utterance batches therefore need no
single batch-wide context and do not combine allocation lifetimes. Model
parameters are shared, but state pools and caches belong to the utterance.
Distinct utterances do not share mutable state storage or retain one another's
arenas. Batch membership has no ownership significance.

The decoder owns the input, the state buffers, and each utterance's
context, and keeps the context alive until every state of that utterance
has been discarded. `Finish` scores only the ended hypotheses without
invalidating their states. Once that utterance's hypotheses and
distributions are released, its arena can be reclaimed wholesale while
other batch members remain live. Pool reuse on pruning within a long input remains possible; a monotonic
arena is not required. Every result referencing arena storage keeps its
context alive. No model-wide reset is needed on individual input completion.
Initially calls complete synchronously; future async batches retain their
input contexts and model resources until completion.

For a neural scorer, `Next` computes and caches the distribution for surviving
states. `Advance` looks up the selected unit's score and creates a deferred
child; the child's recurrent computation waits until its own distribution
is needed. Calls without a preceding `Next` must still return correct scores,
with any necessary parent computation batched internally. Non-neural
backends can implement `Advance` directly without allocating distributions.
`Finish` must account for EOS exactly once in the final ranking and may
need a batched neural step, without mutating reusable states.

The token-to-scorer-unit adapter sits outside this interface: it maps each
emitted recognizer token to scorer IDs and advances through all its units,
summing their scores. Multi-character token candidates may require intermediate
neural computation before token-level pruning; delaying every character step
until a token survives would give incorrect ranking. Identical intermediate
prefixes can share work. Blank and non-emitting CTC repeats never invoke the
adapter. Tokenization, Unicode handling, and partial-boundary state must be
specified by the adapter; scorer methods do not rewrite output tokens.

Initial scope (graehl, 2026-11-12): LMs are trained on exactly the text
convention required by the ASR model. For the common character-LM case,
recognizer pieces contain complete Unicode code points in the same
normalization and casing convention as the LM. Precompute token-to-scorer-unit
sequences at setup, accounting for the recognizer's text representation, and
use those sequences during search. No runtime normalization or case conversion
is needed. This fast path requires compatible vocabularies; do not silently
reinterpret unsupported input representations.

Convenience declarations for lowercasing, case folding, locale, NFC/NFD, and
their operation order are future work, as are general adapters requiring
cross-token Unicode buffering or retokenization. Their eventual support must
not impose per-token transformation work on the matched-text path.

Two extensions of this adapter are recorded in the
[sketches companion](beam-search-and-scorer-fusion.sketches.md): a cost
estimate for a partly consumed scorer token, and lexicon-like vocabulary
mapping.

Search hypotheses retain one state and raw subtotal per scorer,
plus the CTC blank/nonblank sums. The search applies the utterance's
weights when ranking; raw subtotals support diagnostics and retuning,
including when a weight is zero. [Max-alignment mode](beam-search-and-scorer-fusion.md#max-alignment-mode)
recombines on the scorers' state bytes; sum-over-alignments mode does not.

### Required inflight batching (as designed)

The new decoder must support inflight batching: admit new utterances while
others remain active, and finish or cancel individual utterances without
draining the batch. Stable utterance handles identify state independently
of temporary batch positions. Keep each utterance's search state, scorer
contexts, weights, and input progress separate.

At each scheduling step, batch compatible ready decoding/scoring work across
active utterances. Different input lengths and expansion depths must not
force lockstep progress; waiting for more input on one utterance must not
block ready work on another. Preserve result-to-utterance correspondence
when packing and unpacking batches. Bound each scheduling step so admission,
completion, and cancellation can be serviced between steps.

Initially synchronous model/scorer calls with membership changes between
calls satisfy this requirement; concurrent API submissions and overlapping
asynchronous device execution can come later. Release completed or
cancelled inputs independently, after outstanding operations and result
views release their references; never reset another active input's state.
The concrete decoder API and options must expose this lifecycle. Verify
staggered admission, unequal lengths, an input waiting for audio, and removal
while other inputs continue; batched results must agree with independent
decoding under equivalent search settings.

This requirement covers decoder-side scheduling only. Encoder batching,
capture/segmentation, and scheduling across consecutive utterances are
deferred to [encoder-inclusive inflight scheduling](../gaps/sketches/encoder-decoder-inflight-scheduling.md).

### Model and state representation (as designed)

Use C++ abstract interfaces with virtual destructors for scorer models,
following the project's existing interface and ownership conventions.
Each utterance owns a separate refcounted scorer context and
search weights; scorer results remain unweighted so batched requests can
use different weights against the same model.

Scorer state is a fixed-size byte array per hypothesis that the decoder
owns, with its size and optional copy and destroy function pointers
declared in `ScorerInfo` (graehl, 2026-11-13; this replaces a
reference-counted state object per history). An n-gram scorer's context is
the bytes themselves, with no function pointers, allocation, or reference
count. A scorer whose real state is large, variable-length, or
device-resident, such as a recurrent LM carrying what it needs on the
device for an efficient `Next` and `Advance`, wraps a reference to it in
the bytes and sets both functions.

A hypothesis is one record: the model policy's base decoder state followed
directly by every scorer's state bytes (graehl, 2026-11-13; this replaces
the 2026-11-12 choice of separate objects). The base state is whatever that
policy keeps per hypothesis; for CTC, the parent index, last token,
blank-ending and nonblank-ending sums, and per-scorer subtotals. Its size
is fixed for a decoder, and the decoder computes each scorer's offset once
from it and the active scorers' sizes, each rounded up to 8 bytes. A
scorer's state is then the record's address plus a constant, and scoring
touches memory adjacent to the base state. Records live in a contiguous per-utterance pool and
refer to one another by index: a parent link, and a child lookup by parent
index and token. Nothing is keyed or hashed by a record's address.

Try one split from the record, in the first or second version (graehl,
2026-11-13): keep each hypothesis's ranking score in a separate contiguous
array indexed like the pool. Pruning reads only that score across all
active hypotheses, to compute the cutoff and select survivors, so the scan
then touches no record. Keep the split if it measures faster than reading
the score from the records; every other field is used row-wise and stays
in the record.

State bytes are aligned to 8, enough for a pointer, a 64-bit index, or
KenLM's state. `std::max_align_t` is 16 on x86-64 and AArch64 Linux and
would pad every record for a requirement no planned scorer has; a scorer
needing stricter alignment keeps that data in its own storage and puts a
reference in its bytes.

Moving beyond virtual dispatch for scorer models remains an open
micro-optimization, along with parallel arrays in place of records and
function-pointer dispatch that bypasses C++ vtables. The expectation is
that LSTM computation, and to a lesser extent KenLM queries, outweigh this
overhead; that has not been measured for this decoder. Consider these
alternatives if measurements attribute a significant mobile power or
temperature cost to dispatch or layout. Compare sustained power,
temperature, throughput, and recognition quality on the target device; a
smaller object or faster isolated call alone does not establish that
benefit. Check capacity limits on the record index width.

### Decoder lifecycle API (as drafted)

The caller supplies encoded input (CTC frame log-probabilities or
model-specific encoder output); microphone capture and encoder scheduling
remain outside this decoder. A transformer model adapter may require the
complete encoding before it becomes ready.

```cpp
struct UtteranceHandle { uint64_t id; };  // Stable; no reuse within a decoder.
enum class DecodeStatus { Active, Complete, NoAcceptedPath, Cancelled, Error };
enum class DeviceExecution { Alternate, AllowOverlap };

struct ResultOptions {
  size_t nbest = 1;
  bool scores = false;
  bool timestamps = false;
  bool alignment_trace = false;
};

struct DecoderOptions {
  std::vector<ScorerOptions> scorers;  // Same order as the built scorers.
  // One per scorer. Empty means each entry's ScorerOptions::weight.
  std::vector<float> scorer_weights;
  // Shared search options, one family policy, and ExecutionOptions; see
  // "Decode options and pruning direction".
};

struct UtteranceOptions {
  // Empty means DecoderOptions::scorer_weights; otherwise one per scorer.
  std::vector<float> scorer_weights;
  ResultOptions results;
  // Model-compatible search options described below.
};

struct ExecutionOptions {
  DeviceExecution same_device = DeviceExecution::Alternate;
  size_t steps_per_utterance = 1;  // Initial quantum, tunable; not a time bound.
  size_t max_active_utterances;    // Explicit application capacity.
  size_t max_queued_input_bytes;  // Includes retained submitted input buffers.
  // Per-device limits: total accounted memory, transient work, in-flight
  // calls, and per-backend batch item caps. Numeric defaults TBD.
};

class Decoder {
 public:
  // ModelAdapter identifies model family and its post-encoder executors.
  // Scorers and model resources are shared owned objects.
  static std::unique_ptr<Decoder> Create(
      std::shared_ptr<ModelAdapter> model,
      std::vector<std::shared_ptr<Scorer>> scorers,
      const DecoderOptions& options);

  // Capacity refusal is a normal result, not a partially admitted utterance.
  std::optional<UtteranceHandle> TryAdd(const UtteranceOptions& options);
  bool TryAccept(UtteranceHandle id, const EncodedInput& input);
  void InputFinished(UtteranceHandle id);  // No further input; drain then finalize.
  void Cancel(UtteranceHandle id);

  // Retire completed work and schedule a bounded quantum of ready work.
  // May complete synchronous calls; never waits for the whole active set.
  Progress Step();
  void WaitForProgress();  // Explicit wait only for already submitted work.

  DecodeStatus Status(UtteranceHandle id) const;
  DecodeResult GetResult(UtteranceHandle id) const;  // Owned immutable snapshot.
  void Release(UtteranceHandle id);  // Terminal utterances only.
};
```

`ModelAdapter`, `EncodedInput`, `Progress`, and `DecodeResult` are
placeholders for the contracts below. A weight list of the wrong length is a
configuration error at `Create` or `TryAdd`.
Reuse project result/input types where they fit. Decoder methods are called
serially by one coordinator initially; internal device work may overlap.
Application threads enqueue their requests to that coordinator rather than
calling these methods concurrently.

`TryAccept` validates input order, model identity, shapes, and ownership;
accepts the whole supplied chunk or refuses it without consuming any data.
On success the decoder retains the input's backing storage. Frames can be
released only when no live hypothesis or operation needs them. Chunk
boundaries do not finalize an utterance or charge EOS. An input too large
ever to fit a configured limit is an explicit error, not perpetual backpressure.

`Progress` reports advancement, pending operations, and whether ready work
is blocked on input or capacity, so callers can avoid busy polling. A call
to `Step` may block for a synchronous backend batch; the quantum bounds
scheduling work, not a hard wall-clock deadline. `WaitForProgress` must
report when nothing is outstanding rather than wait for unsupplied input.

`InputFinished` permits search and scorer finalization after supplied input
has been consumed. `Complete` means final scoring is done; a hard search
failure yields `NoAcceptedPath`. Backend/runtime errors are reported as
errors, never as infinite-cost hypotheses. A failed shared batch marks all
affected utterances `Error` and reports the exception at the execution
boundary. `Cancel` prevents new work and discards late results but retains
buffers until outstanding operations finish. `Release` drops the decoder's
references; user-held result snapshots remain valid. Invalid or released
handles and unsupported result requests are errors.

`DecodeResult` identifies the utterance and whether the snapshot is final.
Partial output is revisable, with no committed-prefix guarantee initially.
N-best entries contain output token sequences and, when requested, acoustic
and per-scorer raw scores plus the combined ranking score. Timestamp/trace
semantics are model-specific: a representative alignment for a summed CTC
prefix is not its summed-path probability. Define ordering, ties, and whether
N-best uniqueness is by token sequence before implementation. Enable costly
trace retention at admission rather than reconstructing discarded data later.

### Decode options and pruning direction (as designed)

Use shared typed options for common search controls, with model-specific
options for semantics and execution. Keep scorer JSON and factories independent
of which execution loop uses them.

Organize the proposed option structs by responsibility:

| Struct | Fields/concepts to specify | Scope |
|---|---|---|
| `DecoderOptions` | Scorer entries and default weights, shared search options, one compatible family policy, execution options | Decoder construction; validated against model metadata |
| `BeamOptions` | Score gap, minimum/maximum active hypotheses, cutoff slack | Logical retention; each policy defines the counted frontier and enforcement point |
| `CtcSearchOptions` | Sum/max alignment, acoustic-candidate and partial-score pruning; later, auxiliary alignment retention | CTC recurrence and candidate processing |
| `TransducerSearchOptions` | Label-expansion limits and frontier/pruning policy | RNN-T; compatible parts reused for TDT |
| `TdtSearchOptions` | Duration candidates, cross-time frontier ordering and any density limits | Explicit additional TDT policy, still undecided |
| `TransformerSearchOptions` | Token-step beam, EOS/completion rules and output-length limits | Autoregressive decoding |
| `ExecutionOptions` | Admission/input capacity, quantum, alternate/overlap mode, per-device memory/call limits and backend batch caps | Physical work; does not redefine the logical beam |
| `UtteranceOptions` | Scorer weight overrides, result requests, supported per-utterance search overrides | Frozen at admission initially |

Length bonuses must name their unit (ASR token, character, or another
explicit measure) and firing event. Model-dependent defaults must be
resolved and reported before decoding; do not silently inherit Kaldi's
unnormalized beam 16 across models. Do not expose undecided TDT density
controls as working options. Keep field names and the JSON grouping
provisional until the first concrete option structs are reviewed.

For CTC, start from Kaldi FasterDecoder's pruning policy and controls where
applicable: score beam, minimum and maximum active hypotheses, adaptive
cutoffs, and early-rejection slack (`beam_delta`). Reuse that engineering
without importing an FST representation for CTC collapse. Define these
controls on our prefix/state representation rather than promising identical
numeric behavior to an FST-state beam. Sum versus max alignment remains a
separate accumulation choice. In sum mode, cutoffs must account for merged
prefix mass; early rejection of individual alignment contributions can
otherwise remove a prefix that would survive after summation. State any
additional approximation explicitly. Partial character-score pruning and,
later, auxiliary alignment retention have their own separately named budgets.

Separate logical search limits from physical neural-batch capacity. The
latter bounds work items and memory per call; splitting ready work into
physical batches must preserve the policy's merge/prune boundaries and not
silently reduce the logical beam. CTC may still produce highly variable
numbers of alternatives, even though it does not schedule an ASR neural
call for each one.

Anticipate analogous model-specific pruning and batching controls for
RNN-T, TDT, and transformer execution. For TDT, ordering hypotheses by
acoustic time, bounding their density in time buckets, or applying a global
utterance beam are search-policy choices that can change results. Leave
them undecided; specify the scope of each count/score limit before exposing
it. Reject options incompatible with the selected model. Coordinate names
and defaults with the [beam-options gap](../gaps/beam-search-options-and-tuning.md).

Start with one model-policy expansion step per ready utterance as a
scheduling quantum, while leaving room to tune it. For CTC this is one
acoustic timestep; for TDT, successors may advance by different durations.
One step does not bound all internal beam/scorer work. A decoder can queue
ready scorer requests across utterances and compose capacity-bounded calls;
the scorer owns backend packing and batching compatibility. Preserve
backend-owned recurrent states on their device across calls, reusing
workspaces where practical instead of reconstructing histories or copying
states to and from host memory. Only scores required by CPU search need
the `FloatView` host interface. Scorer batching guidance and its exact API
remain to be specified.

### Activation (as proposed)

Use the existing method selector with a new value. The spelling below is
for CTC; the name covering all model families is undecided:

```text
--decoding-method=prefix_beam_search --max-active-paths=8
```

With no scorer configuration, this selects no-LM prefix search. Use
`max-active-paths` for the number of retained prefixes, consistent with
the existing transducer beam-width concept. Model and token-file options
remain those of the chosen CTC recognizer.

Scorers are a `std::vector<ScorerOptions>` configured through
[option collections](json-option-collections.md). One scorer on the
command line is the common case:

```text
--decoding-method=prefix_beam_search --max-active-paths=8 \
  --scorer.type=kenlm --scorer.model=lm.bin --scorer.weight=0.6
```

Several scorers use an index, `--scorer.1.type=onnx-rnn`, or a JSON file:

```text
--decoding-method=prefix_beam_search --max-active-paths=8 \
  --fusion-config=/path/fusion.json
```

Each entry carries `type`, `model`, `unit`, `weight`, `num-threads`, the
existing provider options, and `extra`. Bonus settings are named
separately and their exact controls are open. Library users construct the
same structs directly.

Coordinate common beam controls with
[the beam-options gap](../gaps/beam-search-options-and-tuning.md).
The prototype should establish summed-path correctness on tiny enumerable
examples and preserve it across chunk boundaries before quality comparisons
against the existing greedy path.

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
Until then, anyone who cares tunes `--ctc-score-beam`.

---
slug: lm-states-outlive-their-hypotheses
noticed: 2026-11-18
where: scored-prefix-trie.cc (FreeNode, ReleaseIfUnused), ctc-prefix-search.cc EndFrame, transducer-prefix-search.cc pins, recurrent-lm-scorer.cc
---

# Scorer states live as long as their prefix's descendants

## Observation

Every prefix search (offline and streaming CTC, offline and streaming
transducers, NeMo and TDT) keeps its hypotheses in a `ScoredPrefixTrie`
whose nodes hold each scorer's state bytes. Those bytes are destroyed only
when the trie frees the node (`FreeNode`, or the trie's teardown), and
`ReleaseIfUnused` frees a node only when it is out of the beam (not
pinned) and has no children. A prefix that has left the beam but has a
live descendant therefore keeps its scorer states for as long as that
descendant lives, though nothing will extend it.

For a recurrent LM the state bytes are a reference to an LM node, and an
LM node holds its whole LM step's output (states on the GPU for a GPU
scorer). So every LM node along the path of every live hypothesis back to
the start of the utterance stays alive, each keeping its step's full
output. LM memory then grows with utterance length times the beam, instead
of with the beam. That fits the LSTM pool's in-use peak climbing through a
run (743 MiB on 40 recordings, batches of 8; topic § Execution placement
and batching), though that link is not measured.

## Why it is not trivial

Leaving the beam is not final in the CTC prefix search: a prefix "ab" can
drop out while "abc" lives, and a later frame can extend the surviving "a"
by "b" again, finding the existing node "ab" (`FindOrCreateChild` with
`created == false`) and needing its scorer state again. A transducer
search pins a node once per ring entry (`pins`), so there "unpinned"
already means no hypothesis can arrive at it.

## What is wanted

When a node is unpinned and keeps children, destroy its scorer states and
mark it unscored-for-extension; if a search reaches it again, rescore it
from its parent as a new node (the parent is pinned and so still has its
state), which gives the same scores since scoring is deterministic. The
change belongs in the trie, so every decoder gets it. Measure live LM
nodes and the pool's in-use peak before and after; with a test that a
re-entered prefix scores the same as one never dropped, for each search.

A second, independent waste is granularity: an LM step's output is freed
only with its last node. A slot pool for recurrent states (rows freed
singly) would address it; see the fusion topic's discussion of state
placement.

## Closure

Scorer states of prefixes no search can extend are released at the frame
they leave the beam, in every prefix search; LM memory in use stays
bounded by the beam over a long utterance; transcripts unchanged.

Contributing-model: claude-opus-5-5

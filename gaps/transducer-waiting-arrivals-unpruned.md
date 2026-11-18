---
slug: transducer-waiting-arrivals-unpruned
noticed: 2026-11-18
where: transducer-prefix-search.cc (AddArrival, Prune, EndRound, EndFrame, Outstanding, Finishable)
---

# Transducer arrivals waiting for later frames are cut only by an option

## Observation

The transducer prefix search prunes each queue to `max_active` within the
score beam when it is used: the frontier of a round (`again_`) and the
arrivals for a frame when that frame begins (`EndFrame` → `Prune`). An
arrival for a later frame (`AddArrival`; TDT durations 1 to
`max_duration`) joins that frame's list unpruned and pins its prefix and
scorer states until then. Each round can add up to frontier × token
candidates × durations entries, and a frame has up to
`max_symbols_per_frame` rounds.

`--prune-waiting` (off by default) also cuts every waiting list at each
round end to its best `--waiting-max-active` within `--waiting-score-beam`
of its best (0 for either: `--max-active-paths`, `--score-beam`).

## Measured

600 LibriSpeech test-other recordings, Parakeet TDT 110M (int8, CPU), the
test character LSTM at weight 0.05, `max-active-paths` 4, score beam 12,
4 threads of 2 (2026-11-18):

| | Waiting lists uncut | Cut (4 within 12) |
|---|---:|---:|
| WER | 9.97% (1,018 errors) | 10.04% (1,025) |
| Transcripts changed | | 31 of 600: 11 better, 12 worse |
| LM nodes live at most | 21,041 | 345 |
| Widest LM step | 3,054 nodes | 32 |
| LSTM session's CPU pool held | 128.5 MiB | 4.5 MiB |
| Time | 141.9 s | 120.1 s |

The survival trace (`--trace-survival`, same run) predicted this: the cut
keeps 98.8% of best results' dominant paths while holding 16 entries per
round end on average instead of 2,317, and a cap of 8 within 6 keeps all
of them while holding 22. Waiting entries are mostly longer than their
list's best (19% by 8 or more tokens); arrival pruning keeps entries
within a token or two of the best. The every-path figure (a result charged
for every path that added probability to it) is 0% at almost every
setting and is no guide.

At weight 0.3 the same model gives empty transcripts with this LSTM and
with the character 8-gram alike, so the earlier figures for this gap,
taken there, are withdrawn.

## What is wanted

A decision on the default: turn the cut on for transducer prefix search,
or leave it opt-in. The comparison above shows no WER change within its
noise; a larger set or a second model would narrow that. `Finish` still
scores every waiting entry, which the cut makes few.

## Closure

The default bounds live prefixes and scorer states by a stated policy,
with its measured effect on WER recorded in the fusion topic.

Contributing-model: claude-opus-5-5

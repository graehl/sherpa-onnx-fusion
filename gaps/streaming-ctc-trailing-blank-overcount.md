---
slug: streaming-ctc-trailing-blank-overcount
noticed: 2026-11-12
where: sherpa-onnx/csrc/online-ctc-fst-decoder.cc — DecodeOne
---

**Gap:** Streaming FST CTC decoding overcounts trailing silence across
chunks. Three two-frame all-blank chunks report `num_trailing_blanks` of
2, 6, 12 where 2, 4, 6 is correct, so an endpoint rule can fire before its
silence threshold: six 40-ms frames satisfy a 300-ms rule.

**Noticed while:** Comparing chunked with single-chunk streaming CTC FST
decoding. The fix was deliberately left for later (user-directed, 2026-11-12);
`sherpa-onnx/csrc/online-ctc-fst-decoder-test.cc` records the current
behavior and passes while the defect is present.

## Cause

`DecodeOne` binds `num_trailing_blanks` by reference to the stream's
persistent result, then walks the whole best path from frame 0 on every
chunk and increments that counter for each blank. Blanks from earlier
chunks are therefore counted again on top of the previous total. A nonblank
anywhere in the walk resets the counter, which is why the error appears
only in runs of silence. When `ReachedFinal()` fails for a chunk, the
counter keeps its previous value and does not reflect that chunk at all.

## Fix sketch

Count trailing blanks in a local variable during the walk and assign it to
the result afterwards, so each result reflects exactly the path it was
built from. Decide what a chunk without a final state should report. An
incremental result representation from the
[traceback gap](streaming-ctc-traceback-recomputation.md) must keep the
count correct without a full walk.

## Unaffected path

`--decoding-method=prefix_beam_search` keeps its search across chunks and
does not re-read a traceback. The same test file runs the same input
through it (`AllBlankChunksCountSilenceOnce`) and requires the correct
counts 2, 4, 6. The defect remains in the FST decoder.

## Closure checks

Change the test's oracle to 2, 4, 6, remove its warning and the
`KnownBug` name, and keep its comparison with single-chunk decoding and the
nonblank reset. Check that the endpoint rule in the test no longer fires at
240 ms. Remove this gap and its links in the same commit.

Contributing-model: fable-5.2

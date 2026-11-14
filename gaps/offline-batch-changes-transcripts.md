---
slug: offline-batch-changes-transcripts
noticed: 2026-11-14
where: offline recognizers, feature padding and encoder batching
---

# Decoding several recordings in one call changes some transcripts

## Observation

`sherpa-onnx-offline` puts every file of one call into one padded batch.
For two models the transcripts then differ from decoding each file alone:

- A Vietnamese zipformer transducer (int8 encoder): four recordings per
  call against one gave identical transcripts on 317 of 340 with the
  existing greedy decoder and 307 of 340 with prefix search. The error
  rate was the same within 0.04.
- A NeMo Citrinet CTC model: two of three test recordings changed.

The decoders are given different encoder output, so this is upstream of
the search: the padded frames reach the encoder, or the features, in a
way the model does not fully mask. A correctly masked batch would give
the same output as single decoding.

## Why it matters here

Comparisons in the fusion topic are only like for like when both sides
use the same batching, and a deployment that batches for throughput gets
slightly different transcripts than one that does not.

## Closure

Find where padding leaks (feature padding value, the encoder's length
masking, or the exported model) for one affected model; fix it where the
fork controls it, or record which exported models are affected and decode
them unbatched. Evidence: identical transcripts for a batch and for the
same recordings decoded alone.

Contributing-model: fable-5.2

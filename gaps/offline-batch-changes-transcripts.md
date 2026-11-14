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

## What was found (2026-11-14)

Two causes, both in how the model is run, neither in the search:

- **Padding reaches the model.** With the fp32 Citrinet CTC model, a batch
  of 24 LibriSpeech recordings against the same recordings one at a time
  changes 3 of 24 transcripts. The model does not fully mask padded
  frames.
- **Dynamically quantized (int8) models depend on what else is in the
  batch.** The int8 Citrinet model changes 7 of 24. For the int8 zipformer
  encoder, the longest recording of each batch of four, which has no
  padding at all, changed in 9 of 88 cases, more often than the padded
  ones (14 of 252), and only 8 of the 23 changed transcripts first differ
  in their last fifth. Dynamic quantization takes its value ranges from
  the whole input tensor, so one utterance's numbers depend on its batch
  neighbours.

Neither can be fixed in the decoders. The second is a property of the
exported model; the first needs masking in the export or a padding value
the model was trained to ignore.

## Why it matters here

Comparisons in the fusion topic are only like for like when both sides
use the same batching, and a deployment that batches for throughput gets
slightly different transcripts than one that does not.

## Closure

For comparisons, decode one recording per call, as the reproduction in the
fusion topic does. To close: identify, per model family the fork cares
about, whether an fp32 export with correct length masking gives identical
batched and single transcripts, and document which exported models must
be decoded unbatched. Evidence: identical transcripts for a batch and for
the same recordings decoded alone.

Contributing-model: fable-5.2

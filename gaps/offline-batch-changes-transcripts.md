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

## A batch of 8, with and without padding (2026-11-14)

An export of the public GigaAM v3 RNN-T encoder was run on 8 recordings
twice: with every recording cut to the same length, so the batch has no
padding, and with mixed lengths padded as the recognizer pads. "Changed"
compares each utterance's greedy output in the batch with its output
alone.

| Precision | Equal lengths, no padding | Mixed lengths, padded |
|---|---|---|
| fp32 | identical | relative RMS difference 0.22; 3 of 8 change |
| int8 | relative RMS 0.066; 1 of 8 changes | 0.23; 3 of 8 change |
| int8, MatMul only | relative RMS 0.046; 0 of 8 change | 0.22; 4 of 8 change |

- **In fp32, batching itself is exact.** The encoder takes lengths, but
  its output for the valid frames still depends on the padding, so it is
  not safe to batch with padding as exported. A w2v-BERT CTC export that
  is not public behaved the same way, with no length input at all.
- **Dynamic quantization drift is not small**, and it appears with no
  padding. Its effect on error rate was not measured here.
- The decoders are not involved: with the valid encoder frames held fixed
  and the padded tail overwritten with NaN, infinity or a huge value, no
  decoder's result changed (CTC prefix search in a unit test; the existing
  and prefix transducer decoders on three real models).

The probe was a scratch program on the fork's C++ API, not a retained
test.

## Why it matters here

Comparisons in the fusion topic are only like for like when both sides
use the same batching, and a deployment that batches for throughput gets
slightly different transcripts than one that does not.

## Closure

For comparisons, decode one recording per call. To close: identify, per model family the fork cares
about, whether an fp32 export with correct length masking gives identical
batched and single transcripts, and document which exported models must
be decoded unbatched. Evidence: identical transcripts for a batch and for
the same recordings decoded alone.

An export can take `input_lengths` (int64, valid positions on the input's
sequence axis) and mask padding inside the graph, so that a padded fp32
batch equals each utterance alone. The w2v-BERT model class runs padded
batches when the export has that input and keeps one utterance per call
otherwise. Only its plumbing is checked so far, with a stand-in model
whose output depends on the same frame alone (three recordings of
different lengths, batched and single, identical); the evidence above is
still owed for a real masked export.

Contributing-model: fable-5.2

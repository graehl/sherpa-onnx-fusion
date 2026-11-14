---
slug: litert-model-backend
noticed: 2026-11-14
where: model execution backends, offline and online recognizers, Android builds
---

# Running models through LiteRT

## Intended outcome

User-directed (graehl, 2026-11-14): run recognizer models in LiteRT's
`.tflite` form, without going through ONNX, so that a phone's accelerator
can be used where LiteRT has a delegate for it and ONNX Runtime has no
execution provider. The interest is the Pixel's Tensor processor; the same
path would serve other phones LiteRT accelerates.

The first split is by cost and by what the search needs:

- **In LiteRT:** the encoder, up to its final hidden states, or up to the
  logits where the head is part of the same graph (a CTC model's output
  layer makes the graph depend on the vocabulary size).
- **Outside it at first:** everything after: a CTC model's log-softmax and
  the search; a transducer's prediction network and joiner; the scorers.
  These stay on the CPU or in ONNX Runtime as they are now.
- **Eventually, perhaps all of it** in LiteRT, if the small per-step models
  (joiner, prediction network, recurrent scorer) are worth moving.

## Why the encoder first

The encoder is nearly all of the acoustic model's compute and runs once per
utterance or chunk with no dependence on the search. The parts the search
calls per frame and per hypothesis are small, are called many times, and
have shapes that change with the beam, which accelerators and their
delegates handle poorly. The fork's search already takes encoder output as
plain arrays (`OfflineRecognizer::EncodeStreams` for CTC; the transducer
decoders take the encoder tensor), so the encoder can come from another
runtime without the search knowing.

## Design notes

- **Where it fits.** sherpa-onnx already has per-backend model classes
  beside the ONNX ones (RKNN, QNN, Axera, Ascend), selected by provider
  and model configuration. A LiteRT backend is another of those: a model
  class that produces the same encoder output, behind the same recognizer
  implementations, built only when a CMake option is on.
- **Fixed shapes.** Accelerator delegates generally want static input
  shapes. A whole-utterance encoder then needs either a set of exported
  lengths with padding to the next one, or chunked processing. Padding
  brings the masking question in
  [batch changes transcripts](../offline-batch-changes-transcripts.md):
  the export has to take a length and mask with it, or the padded result
  differs from the unpadded one.
- **Quantization.** Accelerators generally require integer models with
  fixed (static) ranges. That is a different thing from the dynamic int8
  quantization measured so far, and it would remove the dependence of one
  utterance's output on its batch neighbours. Its accuracy has to be
  measured per model.
- **Conversion belongs with the model's export**, in the repository that
  trains it. This repository takes the `.tflite` file and its small
  description (feature options, output kind, frame rate, token table).
- **What is unverified.** Whether a public LiteRT delegate reaches the
  Pixel's Tensor processor is not established: Google's documentation
  listed it as upcoming, with an experimental SDK behind a sign-up, when
  last checked (2026-11-14). The first step is to confirm an available
  delegate on the device before building anything here. Without one,
  LiteRT on the CPU (XNNPACK) is only an alternative to ONNX Runtime on the
  CPU, to be justified by a measurement.
- **Operators.** The encoders use attention with relative position terms,
  layer normalization variants, and convolutions; a delegate that lacks one
  operator sends that part of the graph back to the CPU and can lose the
  gain. Operator coverage is checked per model with the delegate's own
  report, not assumed.

## Work, in order

1. On the device, confirm which delegate is available and run a converted
   encoder with it: time, memory, and how much of the graph it accepted.
2. Compare the converted encoder's output with the ONNX encoder's on the
   same audio, and the error rate through the unchanged search.
3. Add the backend here: build option, model class for the encoder, the
   description file, and the recognizer branch; CTC first, since its search
   needs only the encoder's log probabilities.
4. Transducers: encoder in LiteRT, prediction network and joiner in ONNX
   Runtime.
5. Decide from measurements whether the per-step models should move.

## Completion evidence

On a phone with an accelerator delegate: the same transcripts, within a
stated tolerance, as the ONNX path on a test set; encoder time and energy
against the ONNX CPU path; and the fraction of the encoder the delegate
ran.

Contributing-model: fable-5.2

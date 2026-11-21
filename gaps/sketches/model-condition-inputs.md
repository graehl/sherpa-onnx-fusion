---
slug: model-condition-inputs
noticed: 2026-11-21
where: model wrappers and configs for the recognizer families we export, the recognizer facades, bindings
---

# Recognizer inputs that describe the audio's condition

## Intended outcome

User-directed (graehl, 2026-11-21): support recognizer models that take,
beside the audio features, inputs describing the condition the audio
arrived in. The first is a band label, narrowband (telephone-band, about
4 kHz) or wideband, which the research program now requires as an
encoder input whenever a model is trained on mixed-bandwidth audio
(`~/draft/research/speech-recognition/_PROGRAM/training.md`, band-label
rule). Later candidates from the same line of work are a per-frequency-band
estimate of how much information the audio holds, and a description
carried across the utterances of a session
(`~/draft/research/speech-recognition/gaps/sketches/phone-audio-defect-model.md`).

A default is acceptable (graehl): in most practical use the value is
"wide".

## Two ways to serve it

1. **A constant compiled into the export** (graehl's suggestion). The
   export binds the condition input to its default, so the graph has the
   family's ordinary inputs and this library needs no change. If training
   fixes the wideband vector at zero, as the research rule proposes, the
   wideband export is the plain encoder. A narrowband product would be a
   second export with the other constant. This costs nothing here and is
   enough until a caller needs to choose per utterance.
2. **A run-time input with a default.** The exported encoder keeps the
   extra input; its name, type, shape and default value are recorded in
   the model's metadata. A model wrapper that finds such metadata feeds
   the default unless the caller supplies a value. Models without the
   metadata behave as they do now.

## What the run-time form needs

- **Per-utterance values, not per-recognizer.** The band can differ
  between streams of one recognizer (a headset joins mid-session), so the
  value belongs with the stream or the decode call, and a batch carries
  one value per row. SenseVoice's `language` and `text_norm` inputs are
  the nearest existing case; they are set per recognizer config.
- **A generic carrier.** A named map of small tensors (name to integer or
  float vector) on the stream covers the band label and the later
  per-band description without a new field per condition. Unknown names
  are an error for a model whose metadata does not declare them.
- **Additive edits.** New files where possible, and small additions in
  the model wrappers of the families we export (NeMo transducer and CTC,
  zipformer); no change to families we do not train
  ([FORK.md](../../FORK.md), contribute features, not refactors).
- **Bindings.** The C API and Python first; the others when a consumer
  needs them ([fork feature bindings](../fork-feature-bindings.md)).
- **Who measures the band.** Not this library at first: the caller (the
  app pipeline, which knows the audio route) supplies it. A measurement
  from the features could be added later.

## Unresolved

- Whether any consumer needs the run-time form before a narrowband export
  would do.
- Whether accelerator backends (LiteRT, QNN) accept a second small input
  without losing delegation; a constant export avoids the question.
- Session-carried descriptions imply state between utterances, which is
  the coordinator's in the app pipeline, not the recognizer's.

---
slug: separate-encode-for-other-model-families
noticed: 2026-11-15
where: offline recognizer implementations other than CTC (EncodeStreams, DecodeStreams), OfflineStream
---

# Separate model run and search for families other than CTC

## Intended outcome

User-directed (graehl, 2026-11-15): find what it would take for the other
offline decoders to run the model and the search as separate calls, as CTC
does, so `sherpa-onnx-offline-parallel --decode-threads` and any later
scheduler can size the two apart.

`OfflineRecognizer::EncodeStreams` does something only for CTC. For every
other family it is empty and `DecodeStreams` runs everything, so search
threads also run the model.

## What it takes

**Transducers (zipformer and NeMo): straightforward.** Both implementations
already have the two halves in one function: pad the features, run the
encoder, then hand the encoder output and lengths to the decoder object
(`offline-recognizer-transducer-impl.h`, `-transducer-nemo-impl.h`).

- `EncodeStreams` runs the encoder on the batch and keeps each stream's
  valid frames on the stream. `OfflineStream` already stores that for CTC
  (`SetEncoderOutput`, `HasEncoderOutput`, `TakeEncoderOutput`: a float
  vector and a frame count); nothing in it is CTC-specific.
- `DecodeStreams` pads the kept outputs of the streams it is given into
  one tensor with their lengths, calls the decoder object as now, and runs
  the encoder itself for streams that have none.
- The CTC implementation's two helpers that do the keeping and the
  re-padding (`KeepEncoderOutput`, `DecodeEncodedStreams`) move to a shared
  place; the layout of the encoder output differs for NeMo and needs its
  own slicing.

Roughly a hundred lines and a test like the CTC one (same result whether
or not the two calls are separate, and with different groupings). Cost per
utterance: the encoder output is copied twice more, out of the batch and
back into one.

**Encoder-decoder models with attention (Whisper, Moonshine, Canary,
FireRed, Cohere): possible, less natural.** They run one utterance at a
time, and what the encoder leaves is a set of cross-attention tensors, not
one frames-by-features array, so the stream would have to hold model
tensors. Their second half is a neural decoder, not a CPU search, so the
case for different thread counts is weaker.

**Single-pass models (Paraformer, SenseVoice and similar):** there is no
second step worth separating.

## Would it raise throughput

Not on CPU alone: threads that each run the model and search already scale
there. It pays where the two halves want different concurrency, which is
an encoder on a GPU: one thread encoding wide batches while several
threads search. For CTC with scorers that was measured at 1.65 times with
four search threads
([sketch](../../topics/beam-search-and-scorer-fusion.sketches.md)).

Two things specific to transducers decide whether the same holds:

- The search calls the prediction network and joiner once per frame or
  symbol. With the whole recognizer on one provider those calls go to the
  GPU too and queue behind the encoder. Running the encoder on the GPU
  and the two small networks on the CPU is not something the
  configuration can say today.
- The fork's transducer prefix search runs one utterance at a time, so a
  search thread does not batch scorer calls across a batch as CTC does.

## Closure

A transducer run with `--decode-threads` shows the model threads calling
only the encoder and the search threads only searching, with the same
transcripts as one thread, and a timing with the encoder on a GPU says
whether it helps; or the timing shows no gain and this is dropped.

Contributing-model: fable-5.2

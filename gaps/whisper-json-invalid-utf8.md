---
slug: whisper-json-invalid-utf8
noticed: 2026-11-16
where: sherpa-onnx/csrc/offline-stream.cc, OfflineRecognitionResult::AsJsonString
---

Whisper token strings can contain incomplete UTF-8 byte sequences. When
serialized individually in the `tokens` array, they can make the JSON
output invalid UTF-8 even when the assembled `text` is valid. A Korean
decode through `sherpa-onnx-offline-parallel` exposed this: a strict UTF-8
reader rejected the output before it could extract the transcript.

Consumers that only need the transcript can preserve undecodable bytes
while parsing, then require that `text` encodes strictly as UTF-8. That
workaround does not repair the producer's JSON contract.

Choose a representation for token bytes that preserves token identity
without emitting malformed Unicode. Check existing JSON consumers before
changing the representation; replacing bytes with the replacement
character loses information. Retain a regression case with a multibyte
character split across tokens, plus valid assembled transcript text.

Contributing-model: 6-astra

# sherpa-onnx/python/tests/test_fusion.py
#
# Copyright (c)  2026  graehl
#
# prefix_beam_search with fused scorers and options_json through the Python
# API. Skipped unless SHERPA_ONNX_FUSION_MODELS names a directory holding
# the public models at the paths below.
#
# To run this single test, use
#
#  ctest --verbose -R test_fusion_py

import os
import unittest
import wave
from pathlib import Path
from typing import List, Tuple

import sherpa_onnx

MODELS = os.environ.get("SHERPA_ONNX_FUSION_MODELS", "")
_models = Path(MODELS)
CTC_DIR = _models / "zipformer-ctc-en/sherpa-onnx-zipformer-ctc-en-2023-10-02"
LM = _models / "en-char-6gram/en-librispeech-char-6gram.bin"


def read_wave(filename: str) -> Tuple[List[float], int]:
    with wave.open(filename) as f:
        assert f.getnchannels() == 1, f.getnchannels()
        assert f.getsampwidth() == 2, f.getsampwidth()
        data = f.readframes(f.getnframes())
        rate = f.getframerate()
    samples = memoryview(data).cast("h")
    return [s / 32768 for s in samples], rate


def have(*paths: Path) -> bool:
    return bool(MODELS) and all(p.is_file() for p in paths)


@unittest.skipUnless(
    have(CTC_DIR / "model.int8.onnx", CTC_DIR / "tokens.txt", LM),
    f"needs {CTC_DIR} and {LM}",
)
class TestOfflineFusion(unittest.TestCase):
    wav = str(CTC_DIR / "test_wavs/1.wav")

    def recognizer(self, **kwargs) -> sherpa_onnx.OfflineRecognizer:
        return sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
            model=str(CTC_DIR / "model.int8.onnx"),
            tokens=str(CTC_DIR / "tokens.txt"),
            num_threads=2,
            **kwargs,
        )

    def decode(self, recognizer: sherpa_onnx.OfflineRecognizer) -> str:
        samples, rate = read_wave(self.wav)
        s = recognizer.create_stream()
        s.accept_waveform(rate, samples)
        recognizer.decode_stream(s)
        return s.result.text

    def lm_fusion(self) -> sherpa_onnx.FusionConfig:
        return sherpa_onnx.FusionConfig(
            scorers=[sherpa_onnx.ScorerConfig(type="kenlm", model=str(LM), weight=0.6)]
        )

    def test_prefix_search_without_scorer_matches_greedy(self):
        greedy = self.decode(self.recognizer())
        prefix = self.decode(self.recognizer(decoding_method="prefix_beam_search"))
        self.assertTrue(greedy)
        self.assertEqual(prefix, greedy)

    def test_typed_and_json_scorer(self):
        no_lm = self.decode(self.recognizer(decoding_method="prefix_beam_search"))
        typed = self.decode(
            self.recognizer(
                decoding_method="prefix_beam_search", fusion=self.lm_fusion()
            )
        )
        self.assertTrue(typed)
        heavy = self.decode(
            self.recognizer(
                decoding_method="prefix_beam_search",
                fusion=self.lm_fusion(),
                options_json='{"scorer": [{"weight": 40}], "score-beam": 0}',
            )
        )
        self.assertNotEqual(heavy, no_lm)

    def test_bad_options_json_fails_creation(self):
        with self.assertRaises(ValueError) as e:
            self.recognizer(
                decoding_method="prefix_beam_search",
                options_json='{"no-such-option": 1}',
            )
        self.assertIn("options_json", str(e.exception))


if __name__ == "__main__":
    unittest.main()

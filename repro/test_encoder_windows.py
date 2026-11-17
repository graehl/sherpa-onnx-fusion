"""Checks that silence-directed splitting preserves all samples."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from compare_encoder_windows import silence_split_files


class SilenceSplitTests(unittest.TestCase):
    def check_split(self, audio: np.ndarray) -> list[int]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            sf.write(source, audio, 1000, subtype="PCM_16")
            files, cuts = silence_split_files(source, 20, root / "pieces")
            restored = np.concatenate([sf.read(path)[0] for path in files])
            np.testing.assert_array_equal(restored, sf.read(source)[0])
            self.assertEqual(cuts[0], 0)
            self.assertEqual(cuts[-1], len(audio))
            self.assertTrue(all(0 < b - a <= 20000 for a, b in zip(cuts, cuts[1:])))
            return cuts

    def test_cut_moves_into_pause(self) -> None:
        audio = np.full(25000, 0.5)
        audio[17900:18100] = 0
        self.assertEqual(self.check_split(audio), [0, 18000, 25000])

    def test_short_recording_stays_whole(self) -> None:
        self.assertEqual(self.check_split(np.full(5000, 0.5)), [0, 5000])

    def test_energy_window_at_recording_end(self) -> None:
        self.check_split(np.full(20001, 0.5))


if __name__ == "__main__":
    unittest.main()

"""No-model checks for pinned downloads and the parallel check's failure paths."""

import contextlib
import hashlib
import io
import json
import pathlib
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

import check_parallel
import repro_claims


class ReproTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory(prefix="repro-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = pathlib.Path(self.scratch.name)
        patch = mock.patch.object(repro_claims, "CACHE", self.root / "cache")
        patch.start()
        self.addCleanup(patch.stop)

    def artifact(self, name: str, contents: bytes) -> dict:
        source = self.root / name
        source.write_bytes(contents)
        return {"url": source.as_uri(), "sha256": hashlib.sha256(contents).hexdigest()}

    def test_changed_pin_does_not_reuse_old_payload(self) -> None:
        first = self.artifact("model.bin", b"first")
        old = repro_claims.fetch_artifact("model", first)
        second = self.artifact("model.bin", b"second")
        new = repro_claims.fetch_artifact("model", second)
        self.assertEqual((old / "model.bin").read_bytes(), b"first")
        self.assertEqual((new / "model.bin").read_bytes(), b"second")
        with mock.patch.object(repro_claims.urllib.request, "urlretrieve") as download:
            self.assertEqual(repro_claims.fetch_artifact("model", second), new)
            download.assert_not_called()

    def test_interrupted_download_retries_without_partial_cache(self) -> None:
        spec = self.artifact("model.bin", b"complete")

        def interrupted(url: str, path: pathlib.Path) -> None:
            pathlib.Path(path).write_bytes(b"part")
            raise OSError("connection interrupted")

        with (
            mock.patch.object(repro_claims.urllib.request, "urlretrieve", interrupted),
            self.assertRaisesRegex(OSError, "connection interrupted"),
        ):
            repro_claims.fetch_artifact("model", spec)
        self.assertFalse(repro_claims.artifact_dir("model", spec).exists())
        target = repro_claims.fetch_artifact("model", spec)
        self.assertEqual((target / "model.bin").read_bytes(), b"complete")

    def test_wrong_hash_never_publishes(self) -> None:
        spec = self.artifact("model.bin", b"actual")
        spec["sha256"] = "0" * 64
        with self.assertRaisesRegex(SystemExit, "Artifact hash mismatch"):
            repro_claims.fetch_artifact("model", spec)
        self.assertFalse(repro_claims.artifact_dir("model", spec).exists())

    def test_busy_threads_counts_the_whole_host_and_skips_short_windows(self) -> None:
        # 4 hardware threads; 300 of 800 ticks busy (iowait counts as idle) is 1.5 threads.
        def stats():
            yield "cpu  0 0 0 0 0 0 0 0 0 0\ncpu0 ...\n"
            while True:
                yield "cpu  200 0 100 400 100 0 0 0 50 0\n"

        stat = mock.Mock()
        stat.exists.return_value = True
        stat.read_text.side_effect = stats()
        with (
            mock.patch.object(repro_claims, "PROC_STAT", stat),
            mock.patch.object(repro_claims, "BUSY_WINDOW_SECONDS", 0.01),
            mock.patch.object(repro_claims.os, "cpu_count", return_value=4),
            mock.patch.object(repro_claims.os, "sysconf", return_value=100),
        ):
            with repro_claims.BusyThreads() as busy:
                while stat.read_text.call_count < 4:
                    pass
        self.assertEqual(busy.peak, 1.5)

    def test_result_records_pre_run_identity_and_reconstructible_commands(self) -> None:
        archive = self.root / "dataset.tar"
        with tarfile.open(archive, "w") as tar:
            for name, data in (
                ("sample.trans.txt", b"sample HELLO WORLD\n"),
                ("sample.wav", b""),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        spec = {
            "url": archive.as_uri(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        binary = self.root / "recognizer"
        binary.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' '{\"text\": \"HELLO WORLD\"}'\n"
            "printf '%s\\n' 'Elapsed seconds: 0.125 s' >&2\n"
        )
        binary.chmod(0o755)
        config = {
            "artifacts": {"test": spec},
            "claims": {
                "claim": {
                    "artifacts": ["test"],
                    "test_set": "test",
                    "recognizer": ["--model={test}/model"],
                    "arms": [{"name": "arm", "args": ["--mode=test"]}],
                }
            },
        }
        real_run = subprocess.run
        decoded = False

        def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
            nonlocal decoded
            if argv[0] == "git":
                output = (
                    ("after" if decoded else "before")
                    if "rev-parse" in argv
                    else " M dirty\n"
                )
                return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")
            decoded = True
            return real_run(argv, **kwargs)

        results = self.root / "results"
        with (
            mock.patch.object(repro_claims, "RESULTS", results),
            mock.patch.object(repro_claims.subprocess, "run", run),
            mock.patch.dict("sys.modules", {"soundfile": mock.Mock()}),
        ):
            repro_claims.run_claim(config, "claim", 0, 1, str(binary))
        result = json.loads((results / "claim.json").read_text())
        self.assertEqual(result["fork_commit"], "before")
        self.assertIn("unverified", result["fork_commit_kind"])
        self.assertTrue(result["checkout_dirty"])
        self.assertEqual(
            result["executable"]["sha256"],
            hashlib.sha256(binary.read_bytes()).hexdigest(),
        )
        argv = result["arms"][0]["argv"] + [result["inputs"][0]["wav"]]
        self.assertEqual(argv[0], str(binary))
        self.assertEqual(argv[-2], "--mode=test")
        self.assertTrue(argv[-1].startswith(repro_claims.CACHE_NAME), argv[-1])
        self.assertTrue(pathlib.Path(repro_claims.local(argv[-1])).is_file())
        self.assertEqual(result["arms"][0]["errors"], 0)
        self.assertIn("busy_threads_peak", result["arms"][0])
        self.assertIn("uncontended", result["arms"][0])
        self.assertIn("physical_cores", result["host_threads"])

    def test_parallel_check_rejects_ineffective_pinned_lstm(self) -> None:
        paths = ["sample.wav", check_parallel.MISSING]
        rows = [
            {"index": 0, "wav_filename": paths[0], "text": "same"},
            {"index": 1, "wav_filename": paths[1], "error": "missing"},
        ]
        output = check_parallel.Output(rows, "", 0)
        configurations = {"prefix search": [], "prefix search + character LSTM": []}
        effects = (
            check_parallel.Effect("prefix search + character LSTM", "prefix search"),
        )
        with (
            mock.patch.object(check_parallel, "run_program", return_value=output),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            failures = check_parallel.check(
                pathlib.Path("unused"), "test", [], paths, configurations, effects
            )
        self.assertEqual(failures, 1)
        changed = check_parallel.Output(
            [{**rows[0], "text": "different"}, rows[1]], "", 0
        )
        runs_per_configuration = (
            1 + len(check_parallel.MODES) + len(check_parallel.ROWS_ONLY)
        )
        with (
            mock.patch.object(
                check_parallel,
                "run_program",
                side_effect=[output] * runs_per_configuration
                + [changed] * runs_per_configuration,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            failures = check_parallel.check(
                pathlib.Path("unused"), "test", [], paths, configurations, effects
            )
        self.assertEqual(failures, 0)


if __name__ == "__main__":
    unittest.main()

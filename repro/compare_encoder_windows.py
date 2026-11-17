# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow", "soundfile", "rapidfuzz"]
# ///
"""Compare encoder window lengths on reconstructed TED-LIUM validation talks."""

import argparse
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
import sys

import pyarrow.parquet as pq
import soundfile as sf
from rapidfuzz.distance import Levenshtein
from repro_claims import file_sha256, words

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA256 = "239c4a2874de535f3be7d498bf98abbe7db74512d00a92dda8e14e05c86ade40"


def transcript_words(text: str) -> list[str]:
    text = re.sub(r"<[^>]*>|\[[^]]*\]", " ", text.lower())
    return words(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ctc", "transducer"], default="ctc")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--windows", default="20,15,10,0")
    parser.add_argument(
        "--data", type=Path, required=True, help="Pinned validation parquet"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "build/bin/sherpa-onnx-offline"
    )
    parser.add_argument("--provider", default="cuda")
    args = parser.parse_args()
    if file_sha256(args.data) != DATA_SHA256:
        parser.error(
            "Expected pinned distil-whisper/tedlium-long-form validation parquet"
        )
    windows = list(map(int, args.windows.split(",")))
    if args.limit < 1 or any(w < 0 or 0 < w <= 3 for w in windows):
        parser.error(
            "Positive --limit and windows of 0 or greater than 3 seconds required"
        )
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    rows = pq.read_table(args.data).to_pylist()
    records = []
    for row in rows:
        wav = outdir / (row["speaker_id"] + ".wav")
        if not wav.exists():
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
            sf.write(wav, audio, rate, subtype="PCM_16")
        info = sf.info(wav)
        records.append(
            dict(
                id=row["speaker_id"],
                wav=str(wav),
                seconds=info.duration,
                reference=row["text"],
            )
        )
    records.sort(key=lambda row: row["seconds"])
    (outdir / "manifest.json").write_text(json.dumps(records, indent=2))
    model = args.model_dir.resolve()
    if args.model == "ctc":
        model_args = [f"--zipformer-ctc-model={model}/model.onnx"]
    else:
        model_args = [
            f"--{part}={model}/{part}-epoch-99-avg-1.onnx"
            for part in ["encoder", "decoder", "joiner"]
        ]
    meta = dict(
        command=sys.argv,
        sha=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        binary_sha256=file_sha256(args.binary),
        data_sha256=DATA_SHA256,
        model_files={
            p.name: file_sha256(p)
            for p in model.iterdir()
            if p.is_file() and (".onnx" in p.name or p.name == "tokens.txt")
        },
        normalization="remove angle/square annotations, then repro_claims.words: ASCII letters and apostrophes, case-insensitive",
        condition="reconstructed TED-LIUM validation talks, shortest first; single call per talk, greedy, two CPU threads; fresh process per call, diagnostic timing",
        load_start=os.getloadavg(),
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    for row in records[: args.limit]:
        for window in windows:
            stem = outdir / f"{row['id']}-w{window}"
            result = stem.with_suffix(".json")
            command = [
                str(args.binary.resolve()),
                "--print-args=false",
                *model_args,
                f"--tokens={model}/tokens.txt",
                f"--provider={args.provider}",
                "--num-threads=2",
                "--decoding-method=greedy_search",
                f"--encoder.window.seconds={window}",
                "--encoder.window.overlap=3",
                "--encoder.window.slack=1",
                row["wav"],
            ]
            print(
                f"[decode] {args.model} {row['id']} {row['seconds']:.1f}s window={window}",
                flush=True,
            )
            start = time.monotonic()
            with (
                stem.with_suffix(".out").open("w") as stdout,
                stem.with_suffix(".err").open("w") as stderr,
            ):
                try:
                    done = subprocess.run(
                        command, stdout=stdout, stderr=stderr, timeout=600
                    )
                    code = done.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            record = dict(
                id=row["id"],
                window=window,
                seconds=row["seconds"],
                command=command,
                exit=code,
                wall_seconds=time.monotonic() - start,
                load=os.getloadavg(),
            )
            if code == 0:
                stdout = stem.with_suffix(".out").read_text()
                stderr = stem.with_suffix(".err").read_text()
                match = re.search(
                    r'^\{.*"text": "((?:[^"\\]|\\.)*)"', stdout, re.MULTILINE
                )
                elapsed = re.search(r"Elapsed seconds: ([0-9.]+)", stderr)
                if not match or not elapsed:
                    raise RuntimeError(f"Missing output or timing in {stem}")
                hypothesis = json.loads('"' + match.group(1) + '"')
                reference_words = transcript_words(row["reference"])
                record.update(
                    hypothesis=hypothesis,
                    reference=row["reference"],
                    errors=Levenshtein.distance(
                        reference_words, transcript_words(hypothesis)
                    ),
                    reference_words=len(reference_words),
                    decode_seconds=float(elapsed.group(1)),
                )
            result.write_text(json.dumps(record, indent=2))
            print(
                "[result] "
                + json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ["command", "hypothesis", "reference"]
                    }
                ),
                flush=True,
            )
            if code != 0:
                raise RuntimeError(f"Decode failed: {stem} exit={code}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["soundfile"]
# ///
"""Reproduce a claim listed in claims.toml: fetch its pinned inputs, decode
its test set once per decoder configuration, and report word error rate.

    repro_claims.py list
    repro_claims.py fetch CLAIM
    repro_claims.py run CLAIM [--sample N] [--jobs J] [--binary PATH]
    repro_claims.py table

`run` decodes one recording per sherpa-onnx-offline call, because decoding
several per call changes some models' output. A full run writes
results/CLAIM.json; a --sample run only prints. Downloads go to
$SHERPA_ONNX_REPRO_CACHE, by default ~/.cache/sherpa-onnx-fusion-repro, never
into the checkout.
"""

import argparse
import concurrent.futures
import datetime
import functools
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
# Downloads and converted audio stay out of the checkout.
CACHE = pathlib.Path(
    os.environ.get("SHERPA_ONNX_REPRO_CACHE")
    or pathlib.Path(os.environ.get("XDG_CACHE_HOME") or pathlib.Path.home() / ".cache")
    / "sherpa-onnx-fusion-repro"
)


def load_claims() -> dict:
    with open(HERE / "claims.toml", "rb") as f:
        return tomllib.load(f)


def artifact_dir(name: str) -> pathlib.Path:
    return CACHE / name


def fetch_artifact(name: str, spec: dict) -> pathlib.Path:
    """Download and unpack one pinned artifact; returns its directory."""
    target = artifact_dir(name)
    if (target / ".complete").exists():
        return target
    if not spec.get("url"):
        sys.exit(f"Artifact not published: {name}: {spec.get('note', 'no URL')}")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / spec["url"].rsplit("/", 1)[1]
    if not archive.exists():
        print(f"[fetch] {name} {spec['url']}", file=sys.stderr)
        urllib.request.urlretrieve(spec["url"], archive)
    digest = hashlib.sha256()
    with open(archive, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256"]:
        sys.exit(f"Artifact hash mismatch: {name}: got {digest.hexdigest()}")
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        archive.unlink()
    (target / ".complete").touch()
    return target


def librispeech_recordings(root: pathlib.Path) -> list[tuple[str, pathlib.Path, str]]:
    """(utterance id, wav path, reference) for a LibriSpeech split, sorted by
    id. LibriSpeech ships FLAC, which sherpa-onnx-offline does not read, so
    each file is converted to WAV beside it on first use."""
    import soundfile

    recordings = []
    for transcript in sorted(root.rglob("*.trans.txt")):
        for line in transcript.read_text().splitlines():
            utterance, reference = line.split(" ", 1)
            flac = transcript.parent / f"{utterance}.flac"
            wav = flac.with_suffix(".wav")
            if not wav.exists():
                samples, rate = soundfile.read(flac, dtype="int16")
                soundfile.write(wav, samples, rate, subtype="PCM_16")
            recordings.append((utterance, wav, reference))
    return recordings


def words(text: str) -> list[str]:
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level edit distance."""
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
            )
        previous = current
    return previous[-1]


def decode_one(binary: str, args: list[str], wav: pathlib.Path) -> tuple[str, float]:
    """One recording through sherpa-onnx-offline: (text, decode seconds)."""
    done = subprocess.run(
        [binary, "--print-args=false", *args, str(wav)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = re.search(r'^\{.*"text": "((?:[^"\\]|\\.)*)"', done.stdout, re.MULTILINE)
    seconds = re.search(r"Elapsed seconds: ([0-9.]+)", done.stderr)
    if done.returncode != 0 or not text or not seconds:
        sys.exit(f"Decode failed: {wav}: {done.stderr[-800:]}")
    return json.loads(f'"{text.group(1)}"'), float(seconds.group(1))


def resolve(args: list[str], dirs: dict[str, pathlib.Path]) -> list[str]:
    """Replace {artifact} in arguments with that artifact's directory."""
    return [a.format(**{k: str(v) for k, v in dirs.items()}) for a in args]


def fetch_claim(config: dict, claim: dict) -> dict[str, pathlib.Path]:
    return {
        name: fetch_artifact(name, config["artifacts"][name])
        for name in claim["artifacts"]
    }


def run_claim(config: dict, name: str, sample: int, jobs: int, binary: str) -> None:
    claim = config["claims"][name]
    dirs = fetch_claim(config, claim)
    recordings = librispeech_recordings(dirs[claim["test_set"]])
    if sample:
        recordings = recordings[:: max(1, len(recordings) // sample)][:sample]
    references = [words(reference) for _, _, reference in recordings]
    total_words = sum(len(r) for r in references)
    arms = []
    for arm in claim["arms"]:
        args = resolve(claim["recognizer"] + arm["args"], dirs)
        with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
            decoded = list(
                pool.map(
                    functools.partial(decode_one, binary, args),
                    [wav for _, wav, _ in recordings],
                )
            )
        errors = sum(
            word_errors(r, words(text)) for r, (text, _) in zip(references, decoded)
        )
        arms.append(
            {
                "name": arm["name"],
                "wer": round(100 * errors / total_words, 2),
                "errors": errors,
                "decode_seconds": round(sum(s for _, s in decoded), 1),
            }
        )
        print(
            f"[arm] {name} {arm['name']} wer={arms[-1]['wer']} errors={errors} "
            f"decode_s={arms[-1]['decode_seconds']}",
            flush=True,
        )
    if sample:
        print(f"[sample] {len(recordings)} recordings; results/{name}.json not written")
        return
    commit = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    RESULTS.mkdir(exist_ok=True)
    result = {
        "claim": name,
        "fork_commit": commit,
        "date": datetime.date.today().isoformat(),
        "host": f"{platform.machine()}, {jobs} decodes at a time",
        "test_set": {
            "name": claim["test_set"],
            "recordings": len(recordings),
            "words": total_words,
        },
        "arms": arms,
        "pins": {a: config["artifacts"][a]["sha256"] for a in claim["artifacts"]},
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")


def print_table(config: dict) -> None:
    """The recorded results as a Markdown table, one row per decoder."""
    print("| Claim | Test set | Decoder | WER % | Decode s |")
    print("|---|---|---|---:|---:|")
    for name in config["claims"]:
        path = RESULTS / f"{name}.json"
        if not path.exists():
            print(f"| {name} | | not run yet | | |")
            continue
        result = json.loads(path.read_text())
        test_set = f"{result['test_set']['name']} ({result['test_set']['recordings']})"
        for arm in result["arms"]:
            print(
                f"| {name} | {test_set} | {arm['name']} | {arm['wer']:.2f} | {arm['decode_seconds']} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("table")
    for command in ("fetch", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("claim")
    run = commands.choices["run"]
    run.add_argument(
        "--sample",
        type=int,
        default=0,
        help="--sample N: decode N recordings spread over the set",
    )
    run.add_argument(
        "--jobs", type=int, default=2, help="--jobs J: decodes running at once"
    )
    run.add_argument(
        "--binary",
        default=str(HERE.parent / "build" / "bin" / "sherpa-onnx-offline"),
        help="--binary PATH: sherpa-onnx-offline built from this checkout",
    )
    options = parser.parse_args()
    config = load_claims()
    if options.command == "list":
        for name, claim in config["claims"].items():
            print(f"{name}: {claim['summary']}")
    elif options.command == "table":
        print_table(config)
    elif options.claim not in config["claims"]:
        sys.exit(f"Unknown claim: {options.claim}; see `repro_claims.py list`")
    elif options.command == "fetch":
        fetch_claim(config, config["claims"][options.claim])
    else:
        run_claim(config, options.claim, options.sample, options.jobs, options.binary)


if __name__ == "__main__":
    main()

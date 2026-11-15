#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Check sherpa-onnx-offline-parallel: threads, search threads and sorting
must not change what it prints.

    check_parallel.py [--build-dir DIR] [--offline] [--lstm-model FILE]

It decodes the three recordings shipped with the pinned English zipformer CTC
model, repeated, with one unreadable file among them. For three recognizers
(that CTC model, a zipformer transducer, and a NeMo Parakeet TDT transducer)
and each search configuration, it compares every way of running the program
against one thread:

- a row for every input, in list order, naming its file;
- an error row for the unreadable file and for no other;
- the same text as one thread, at one recording per batch, where batching
  cannot change a model's output.

The search configurations are greedy search, upstream's beam search for the
transducers, the prefix search, the prefix search with a small n-gram model
written by this script (when the build has KenLM), the prefix search with a
tiny pinned character LSTM made for this check, and with --lstm-model also an
onnx-rnn scorer over the CTC recognizer's tokens. Only pinned public archives
are downloaded, about 600 MB in all; --offline skips, successfully, whatever
has not been fetched.
"""

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
import tempfile

import repro_claims

# The recordings every recognizer decodes: the three shipped with the CTC model.
RECORDINGS = "zipformer-ctc-en"
RECORDINGS_DIR = "sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs"


@dataclasses.dataclass(frozen=True)
class Recognizer:
    name: str
    artifact: str
    flags: tuple[str, ...]  # {dir} is the artifact's directory
    searches: tuple[str, ...]  # names from search_configurations


_ZIPFORMER = "{dir}/sherpa-onnx-zipformer-small-en-2023-06-26"
_TDT = "{dir}/sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8"
RECOGNIZERS = (
    Recognizer(
        "zipformer CTC",
        "zipformer-ctc-en",
        (
            "--zipformer-ctc-model={dir}/sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx",
            "--tokens={dir}/sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt",
        ),
        (
            "greedy",
            "prefix search",
            "prefix search + n-gram",
            "prefix search + character LSTM",
            "prefix search + n-gram + token LSTM",
        ),
    ),
    Recognizer(
        "zipformer transducer",
        "zipformer-small-en",
        (
            f"--encoder={_ZIPFORMER}/encoder-epoch-99-avg-1.int8.onnx",
            f"--decoder={_ZIPFORMER}/decoder-epoch-99-avg-1.int8.onnx",
            f"--joiner={_ZIPFORMER}/joiner-epoch-99-avg-1.int8.onnx",
            f"--tokens={_ZIPFORMER}/tokens.txt",
        ),
        ("greedy", "upstream beam search", "prefix search", "prefix search + n-gram"),
    ),
    Recognizer(
        "Parakeet TDT",
        "parakeet-tdt-110m-en",
        (
            f"--encoder={_TDT}/encoder.int8.onnx",
            f"--decoder={_TDT}/decoder.int8.onnx",
            f"--joiner={_TDT}/joiner.int8.onnx",
            f"--tokens={_TDT}/tokens.txt",
            "--model-type=nemo_transducer",
        ),
        ("greedy", "upstream beam search", "prefix search"),
    ),
)
# A tiny character LSTM made for this check; it is not a usable language
# model. Two artifacts: its step graph and its unit list.
LSTM = "test-char-lstm"
LSTM_FILE = "en-char-lstm-256x1.onnx"
LSTM_UNITS = "test-char-lstm-units"
LSTM_UNITS_FILE = "en-char-lstm-256x1.units.hex"
MISSING = "/nonexistent/check-parallel-missing.wav"
MISSING_AT = 2  # position of the unreadable file in the list
REPEATS = 2  # times each shipped recording appears; more only costs time

# How to run the program, each compared with one thread. All use one
# recording per batch; sorting then only changes the order of decoding.
MODES: dict[str, list[str]] = {
    "4 threads": ["--nj=4"],
    "2 model threads, 3 search threads": [
        "--nj=2",
        "--decode-threads=3",
        "--decode-queue-size=1",
    ],
    "sorted by length, 3 threads": ["--nj=3", "--sort-by-length=true"],
    "sorted, 1 model thread, 2 search threads": [
        "--nj=1",
        "--sort-by-length=true",
        "--decode-threads=2",
    ],
}
# Batches of several recordings may change a model's output, so this one is
# checked for its rows only.
BATCHED = ["--nj=2", "--batch-size=4", "--sort-by-length=true", "--decode-threads=2"]


def write_arpa(path: pathlib.Path) -> None:
    """A small character bigram model: enough to steer the search, and to
    make the n-gram scorer do real lookups. Its quality is beside the point."""
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["'", "▁"]
    pairs = ["t h", "h e", "e ▁", "▁ t", "a n", "n d", "d ▁", "▁ a", "i n", "e r"]
    lines = ["\\data\\", f"ngram 1={len(letters) + 3}", f"ngram 2={len(pairs)}", ""]
    lines += ["\\1-grams:", "-2.0\t<unk>", "-99\t<s>\t-0.3", "-1.2\t</s>"]
    lines += [f"{-1.0 - 0.03 * i:.2f}\t{c}\t-0.2" for i, c in enumerate(letters)]
    lines += ["", "\\2-grams:"] + [f"-0.3\t{pair}" for pair in pairs]
    lines += ["", "\\end\\", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


@dataclasses.dataclass
class Output:
    rows: list[dict]
    stderr: str
    returncode: int


def run_program(binary: pathlib.Path, args: list[str]) -> Output:
    done = subprocess.run(
        [str(binary), "--print-args=false", "--pause-seconds=0", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    rows = []
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    return Output(rows, done.stderr, done.returncode)


def row_problems(output: Output, paths: list[str]) -> list[str]:
    """What is wrong with the rows of one run, apart from their text."""
    if output.returncode != 0:
        return [f"exit status {output.returncode}: {output.stderr[-400:]}"]
    if len(output.rows) != len(paths):
        return [f"{len(output.rows)} rows for {len(paths)} inputs"]
    problems = []
    for i, (row, path) in enumerate(zip(output.rows, paths)):
        if row.get("index") != i or row.get("wav_filename") != path:
            problems.append(f"row {i} is index {row.get('index')}")
        elif ("error" in row) != (path == MISSING):
            problems.append(f"row {i}: error row expected only for the missing file")
        elif path != MISSING and "text" not in row:
            problems.append(f"row {i} has no text")
    return problems


def texts(output: Output) -> list[str | None]:
    return [row.get("text") for row in output.rows]


def search_configurations(
    arpa: pathlib.Path,
    char_lstm: tuple[pathlib.Path, pathlib.Path] | None,
    lstm_model: str | None,
) -> dict[str, list[str]]:
    """The decoder flags of each configuration. "char_lstm" is the pinned test
    LSTM's step graph and its unit list, or None to leave it out."""
    kenlm = [
        "--decoding-method=prefix_beam_search",
        "--scorer-text-case=lower",
        "--scorer.0.type=kenlm",
        f"--scorer.0.model={arpa}",
        "--scorer.0.weight=0.3",
    ]
    configurations = {
        "greedy": ["--decoding-method=greedy_search"],
        "upstream beam search": ["--decoding-method=modified_beam_search"],
        "prefix search": ["--decoding-method=prefix_beam_search"],
        "prefix search + n-gram": kenlm,
    }
    if char_lstm:
        extra = json.dumps({"vocab": str(char_lstm[1]), "vocab-encoding": "hex"})
        configurations["prefix search + character LSTM"] = [
            "--decoding-method=prefix_beam_search",
            "--scorer.0.type=onnx-rnn",
            "--scorer.0.unit=char",
            f"--scorer.0.model={char_lstm[0]}",
            f"--scorer.0.extra={extra}",
            # The pinned fixture must change some words of the shipped recordings.
            "--scorer.0.weight=1.0",
        ]
    if lstm_model:
        configurations["prefix search + n-gram + token LSTM"] = kenlm + [
            "--scorer.1.type=onnx-rnn",
            f"--scorer.1.model={lstm_model}",
            "--scorer.1.unit=token",
            "--scorer.1.weight=0.2",
        ]
    return configurations


def check(
    binary: pathlib.Path,
    label: str,
    recognizer: list[str],
    paths: list[str],
    configurations: dict[str, list[str]],
) -> int:
    """Runs one recognizer through every configuration and mode; returns the
    number of failures."""
    failures = 0
    baseline = None
    for name, search in configurations.items():
        configuration = name
        name = f"{label}, {name}"
        reference = run_program(binary, recognizer + search + ["--nj=1"])
        if "Unknown scorer type 'kenlm'" in reference.stderr:
            print(f"[check] SKIP {name}: this build has no KenLM")
            continue
        problems = row_problems(reference, paths)
        if problems:
            print(f"[check] FAIL {name}, 1 thread: {problems[0]}")
            failures += 1
            continue
        if configuration == "prefix search":
            baseline = texts(reference)
        elif configuration == "prefix search + character LSTM" and (
            baseline is None or texts(reference) == baseline
        ):
            print(f"[check] FAIL {name}: pinned LSTM has no verified output effect")
            failures += 1
        runs = [(mode, args, True) for mode, args in MODES.items()]
        runs.append(("batches of 4 (rows only)", BATCHED, False))
        for mode, args, compare_text in runs:
            output = run_program(binary, recognizer + search + args)
            problems = row_problems(output, paths)
            if not problems and compare_text and texts(output) != texts(reference):
                differing = sum(a != b for a, b in zip(texts(output), texts(reference)))
                problems = [f"{differing} texts differ from 1 thread"]
            if problems:
                print(f"[check] FAIL {name}, {mode}: {problems[0]}")
                failures += 1
            else:
                print(f"[check] ok   {name}, {mode}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--build-dir",
        default=str(repro_claims.HERE.parent / "build"),
        help="build directory holding bin/sherpa-onnx-offline-parallel",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not download: skip, successfully, whatever is not fetched",
    )
    parser.add_argument(
        "--lstm-model",
        help="an onnx-rnn model over the CTC recognizer's tokens, to check too",
    )
    args = parser.parse_args()

    binary = pathlib.Path(args.build_dir) / "bin" / "sherpa-onnx-offline-parallel"
    if not binary.exists():
        sys.exit(f"Not built: {binary}")
    artifacts = repro_claims.load_claims()["artifacts"]

    def fetch(name: str) -> pathlib.Path | None:
        """The artifact's directory, or None if --offline and not fetched."""
        complete = repro_claims.artifact_dir(name, artifacts[name]) / ".complete"
        if args.offline and not complete.exists():
            return None
        return repro_claims.fetch_artifact(name, artifacts[name])

    # Every recognizer decodes the recordings shipped with the CTC model.
    recordings = fetch(RECORDINGS)
    if recordings is None:
        print(f"[check] SKIP all: --offline and artifact {RECORDINGS} is not fetched")
        return
    wavs = sorted(str(p) for p in (recordings / RECORDINGS_DIR).glob("*.wav"))
    paths = wavs * REPEATS
    paths.insert(MISSING_AT, MISSING)
    char_lstm = None
    lstm, units = fetch(LSTM), fetch(LSTM_UNITS)
    if lstm is None or units is None:
        print(f"[check] SKIP the character LSTM: --offline and {LSTM} is not fetched")
    else:
        char_lstm = (lstm / LSTM_FILE, units / LSTM_UNITS_FILE)

    failures = 0
    with tempfile.TemporaryDirectory(prefix="check-parallel-") as scratch:
        wav_list = pathlib.Path(scratch) / "wavs.list"
        wav_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
        arpa = pathlib.Path(scratch) / "characters.arpa"
        write_arpa(arpa)
        available = search_configurations(arpa, char_lstm, args.lstm_model)
        for recognizer in RECOGNIZERS:
            directory = fetch(recognizer.artifact)
            if directory is None:
                print(f"[check] SKIP {recognizer.name}: --offline and not fetched")
                continue
            flags = [flag.format(dir=directory) for flag in recognizer.flags]
            flags += ["--num-threads=1", f"--wav-scp={wav_list}"]
            configurations = {
                name: available[name]
                for name in recognizer.searches
                if name in available
            }
            failures += check(binary, recognizer.name, flags, paths, configurations)
    if failures:
        sys.exit(f"[check] {failures} failed")
    print("[check] all passed")


if __name__ == "__main__":
    main()

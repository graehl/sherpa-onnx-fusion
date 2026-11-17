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

- a row for every input naming its file, in list order when that is asked
  for and each index once when rows are printed as they finish;
- an error row for the unreadable file and for no other;
- the same text as one thread, at one recording per model batch, where
  batching cannot change a model's output; this includes search calls of
  three recordings (--search-batch-size), which must not change it either.
  Model batches of four, searched whole, in calls of two and in calls of
  three, are checked for their rows.

The search configurations are greedy search, upstream's beam search for the
transducers, the prefix search, the prefix search with a small n-gram model
written by this script (when the build has KenLM), the prefix search with a
tiny pinned character LSTM made for this check, and with --lstm-model also an
onnx-rnn scorer over the CTC recognizer's tokens. The prefix search also runs
with max-alignment CTC, a narrow beam, no score limit, and, for the zipformer
transducer, modified_beam_search's rules; and greedy search and the n-gram
search run with the encoder in 4 s windows, blended by center weight, cut,
probability (CTC), per-utterance normalization, and uniform weights with
context-only edges.

Some configurations must also have an effect on the shipped recordings at one
thread: the character LSTM and most window settings must change some text,
the narrow beam must on the TDT model, and modified_beam_search's rules must
give upstream's beam search's texts exactly. Only pinned public archives
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
class Effect:
    """What one configuration's one-thread texts must be against another's
    on the shipped recordings: different (an option that must change some
    word) or the same (a decoder that must be reproduced)."""

    configuration: str
    baseline: str
    same: bool = False


@dataclasses.dataclass(frozen=True)
class Recognizer:
    name: str
    artifact: str
    flags: tuple[str, ...]  # {dir} is the artifact's directory
    searches: tuple[str, ...]  # names from search_configurations
    effects: tuple[Effect, ...] = ()


# Encoder windows short enough that every shipped recording is cut.
WINDOWS = ["--encoder.window.seconds=4", "--encoder.window.overlap=1"]
WINDOWED_SEARCHES = (
    "greedy, 4 s windows",
    "greedy, 4 s windows, cut",
    "greedy, 4 s windows, normalized per utterance",
    "greedy, 4 s windows, uniform with context edges",
    "prefix search + n-gram, 4 s windows",
)


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
            "prefix search, max alignment",
            "prefix search, narrow beam",
            "prefix search, no score limit",
            "prefix search + n-gram",
            "prefix search + character LSTM",
            "prefix search + n-gram + token LSTM",
            "greedy, 4 s windows, probability blend",
        )
        + WINDOWED_SEARCHES,
        (
            # The pinned fixture must change some words of the recordings.
            Effect("prefix search + character LSTM", "prefix search"),
            Effect("greedy, 4 s windows", "greedy"),
            Effect("greedy, 4 s windows, cut", "greedy"),
            Effect("greedy, 4 s windows, probability blend", "greedy"),
            Effect("greedy, 4 s windows, uniform with context edges", "greedy"),
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
        (
            "greedy",
            "upstream beam search",
            "prefix search",
            "prefix search, modified_beam_search rules",
            "prefix search, narrow beam",
            "prefix search, no score limit",
            "prefix search + n-gram",
        )
        + WINDOWED_SEARCHES,
        (
            Effect(
                "prefix search, modified_beam_search rules",
                "upstream beam search",
                same=True,
            ),
        ),
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
        (
            "greedy",
            "upstream beam search",
            "prefix search",
            "prefix search, narrow beam",
            "prefix search, no score limit",
        )
        + WINDOWED_SEARCHES,
        (
            Effect("prefix search, narrow beam", "prefix search"),
            Effect("greedy, 4 s windows", "greedy"),
            Effect("greedy, 4 s windows, cut", "greedy"),
            Effect("greedy, 4 s windows, normalized per utterance", "greedy"),
        ),
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
# recording per batch; sorting then only changes the order of decoding. The
# program sorts and prints results as they finish unless told otherwise, so
# every run names both choices; run_program asks for list order by default.
UNSORTED = "--sort-by-length=false"
AS_FINISHED = "--ordered-output=false"
MODES: dict[str, list[str]] = {
    "4 threads": ["--nj=4", UNSORTED],
    "2 model threads, 3 search threads": [
        "--nj=2",
        UNSORTED,
        "--decode-threads=3",
        "--decode-queue-size=1",
    ],
    "sorted by length, 3 threads": ["--nj=3", "--sort-by-length=true"],
    "sorted, 3 threads, printed as finished": [
        "--nj=3",
        "--sort-by-length=true",
        AS_FINISHED,
    ],
    "sorted, 1 model thread, 2 search threads": [
        "--nj=1",
        "--sort-by-length=true",
        "--decode-threads=2",
    ],
    # The model runs one recording per call, as for the reference, and each
    # search call takes three: sharing a search call must change no text.
    "search calls of 3, 2 search threads": [
        "--nj=2",
        UNSORTED,
        "--decode-threads=2",
        "--search-batch-size=3",
    ],
    "search calls of 3, no search threads": [
        "--nj=1",
        UNSORTED,
        "--search-batch-size=3",
    ],
}
# Batches of several recordings may change a model's output, so these are
# checked for their rows only: model batches searched whole, split into
# calls the search reads in place, and split with leftovers joining the
# next batch.
BATCHED = ["--nj=2", "--batch-size=4", "--sort-by-length=true", "--decode-threads=2"]
ROWS_ONLY: dict[str, list[str]] = {
    "batches of 4 (rows only)": BATCHED,
    "batches of 4, search calls of 2 (rows only)": BATCHED
    + ["--search-batch-size=2"],
    "batches of 4, search calls of 3 (rows only)": BATCHED
    + ["--search-batch-size=3"],
}


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
    """Runs the program asking for rows in list order, unless "args" asks
    for them as they finish; those rows are returned sorted by index, so
    that a missing or repeated index still shows as a misplaced row."""
    as_finished = AS_FINISHED in args
    order = [] if as_finished else ["--ordered-output=true"]
    done = subprocess.run(
        [str(binary), "--print-args=false", "--pause-seconds=0", *order, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )
    rows = []
    for line in done.stdout.splitlines():
        if line.startswith("{"):
            rows.append(json.loads(line))
    if as_finished:
        rows.sort(key=lambda row: row.get("index", -1))
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
    greedy = ["--decoding-method=greedy_search"]
    prefix = ["--decoding-method=prefix_beam_search"]
    configurations = {
        "greedy": greedy,
        "upstream beam search": ["--decoding-method=modified_beam_search"],
        "prefix search": prefix,
        "prefix search, max alignment": prefix + ["--ctc-alignment=max"],
        "prefix search, narrow beam": prefix
        + ["--max-active-paths=2", "--token-candidates=2", "--score-beam=3"],
        "prefix search, no score limit": prefix + ["--score-beam=0"],
        "prefix search, modified_beam_search rules": prefix
        + ["--modified-beam-search-rules=true"],
        "prefix search + n-gram": kenlm,
        "greedy, 4 s windows": greedy + WINDOWS,
        "greedy, 4 s windows, cut": greedy + WINDOWS + ["--encoder.window.weight=cut"],
        "greedy, 4 s windows, probability blend": greedy
        + WINDOWS
        + ["--encoder.window.blend=prob"],
        "greedy, 4 s windows, normalized per utterance": greedy
        + WINDOWS
        + ["--encoder.window.normalize=utterance"],
        "greedy, 4 s windows, uniform with context edges": greedy
        + WINDOWS
        + ["--encoder.window.center-power=0", "--encoder.window.context-frames=4"],
        "prefix search + n-gram, 4 s windows": kenlm + WINDOWS,
    }
    if char_lstm:
        extra = json.dumps({"vocab": str(char_lstm[1]), "vocab-encoding": "hex"})
        configurations["prefix search + character LSTM"] = [
            "--decoding-method=prefix_beam_search",
            "--scorer.0.type=onnx-rnn",
            "--scorer.0.unit=char",
            f"--scorer.0.model={char_lstm[0]}",
            f"--scorer.0.extra={extra}",
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


def effect_problem(effect: Effect, one_thread: dict[str, list[str | None]]) -> str:
    """Why the effect does not hold, or "" if it does."""
    baseline = one_thread.get(effect.baseline)
    if baseline is None:
        return f"its baseline '{effect.baseline}' did not run"
    differing = sum(
        a != b for a, b in zip(one_thread[effect.configuration], baseline)
    )
    if effect.same and differing:
        return f"{differing} texts differ from '{effect.baseline}'"
    if not effect.same and not differing:
        return f"no text differs from '{effect.baseline}'"
    return ""


def check(
    binary: pathlib.Path,
    label: str,
    recognizer: list[str],
    paths: list[str],
    configurations: dict[str, list[str]],
    effects: tuple[Effect, ...] = (),
) -> int:
    """Runs one recognizer through every configuration and mode; returns the
    number of failures."""
    failures = 0
    one_thread: dict[str, list[str | None]] = {}
    for name, search in configurations.items():
        configuration = name
        name = f"{label}, {name}"
        reference = run_program(binary, recognizer + search + ["--nj=1", UNSORTED])
        if "Unknown scorer type 'kenlm'" in reference.stderr:
            print(f"[check] SKIP {name}: this build has no KenLM")
            continue
        problems = row_problems(reference, paths)
        if problems:
            print(f"[check] FAIL {name}, 1 thread: {problems[0]}")
            failures += 1
            continue
        one_thread[configuration] = texts(reference)
        for effect in effects:
            if effect.configuration != configuration:
                continue
            problem = effect_problem(effect, one_thread)
            if problem:
                print(f"[check] FAIL {name}: {problem}")
                failures += 1
            else:
                relation = "same as" if effect.same else "differs from"
                print(f"[check] ok   {name}: {relation} '{effect.baseline}'")
        runs = [(mode, args, True) for mode, args in MODES.items()]
        runs += [(mode, args, False) for mode, args in ROWS_ONLY.items()]
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
            effects = tuple(e for e in recognizer.effects if e.configuration in configurations)
            failures += check(
                binary, recognizer.name, flags, paths, configurations, effects
            )
    if failures:
        sys.exit(f"[check] {failures} failed")
    print("[check] all passed")


if __name__ == "__main__":
    main()

# Reproducing the fork's claims

[claims.toml](claims.toml) has recipes for two comparisons in
[FORK.md](../FORK.md): English CTC with the shipped graph, and English CTC
with a character n-gram. Each entry names the recognizer, the test set, and
the decoder configurations compared, with the exact command-line flags.
[repro_claims.py](repro_claims.py) fetches the pinned inputs, decodes the
test set once per configuration, and reports word error rate.

```bash
cd repro
mise install                              # the pinned Python and uv
mise run build                            # sherpa-onnx-offline, with KenLM, into ../build
mise run list                             # the claims
mise run claim ctc-shipped-graph --sample 50   # a few minutes
mise run claim ctc-shipped-graph          # the whole test set; writes results/
mise run table                            # the recorded results
```

Without `mise`: `uv run repro_claims.py run <claim>`, or any Python 3.11
or later with `soundfile` installed.

- **Reading a claim without running it.** `claims.toml` gives the flags of
  every configuration. A full run writes `results/<claim>.json` with the
  test set and size, word errors, decode seconds, input hashes, and command
  arguments. It records the checkout revision and dirty state before the
  run, plus the executable's path and SHA-256; the checkout revision does
  not establish which source built that executable.
- **Where files go.** Downloads and converted audio go to
  `$SHERPA_ONNX_REPRO_CACHE`, by default `~/.cache/sherpa-onnx-fusion-repro`.
  Downloads are keyed by artifact name and SHA-256.
  The claim runner writes into the checkout only under `results/`, after
  a full run; `mise run build` writes separately to `../build`.
- **One recording per call.** Decoding several recordings in one call
  changes some models' output
  ([gap](../gaps/offline-batch-changes-transcripts.md)), so every
  configuration decodes them one at a time. `--jobs` sets how many calls
  run at once; decode seconds are the recognizer's own figure summed, and
  are comparable between configurations only at the same `--jobs` on an
  otherwise idle machine.
- **Whether the timing was uncontended.** On Linux each configuration
  records `busy_threads_peak`: the most hardware threads busy in any
  10-second window while it ran, counting every process on the host. It
  is `uncontended` when that peak never exceeds the host's physical
  cores; `mise run table` shows the verdict. Choose `--jobs` so that two
  threads per call leave room for whatever else the host runs.
- **What is not here.** The public token-LSTM CTC and transducer comparisons
  have no recipes yet. Claims that need unpublished language models also
  have no entries.

## Adding a claim

### Long-recording encoder windows

`compare_encoder_windows.py` compares whole-recording encoding (window 0)
with 20, 15 and 10 s overlapping windows, using greedy decoding, 3 s
overlap and 1 s slack. It accepts the validation parquet from
[`distil-whisper/tedlium-long-form`](https://huggingface.co/datasets/distil-whisper/tedlium-long-form),
file `data/validation-00000-of-00001-9ed099229d0cbe10.parquet`, pinned to
SHA-256 `239c4a2874de535f3be7d498bf98abbe7db74512d00a92dda8e14e05c86ade40`.
These eight talks were reconstructed by concatenating the original
transcribed segments; untranscribed gaps are absent. They are development
evidence, not an untouched full-recording test set.

```bash
uv run repro/compare_encoder_windows.py --data validation.parquet \
  --model ctc --model-dir /path/to/sherpa-onnx-zipformer-ctc-en-2023-10-02 \
  --output /path/to/new-results --provider cuda
```

For the small English zipformer transducer, use `--model transducer`
and its model directory. The pinned model archives are in `claims.toml`.
Run under a memory cap, particularly for whole-recording encoding. The
output directory must be new; results include every hypothesis/reference,
word-edit count, decode time, command, model hashes and executable hash.
Normalization removes bracketed annotations and uses the existing recipe's
case-insensitive ASCII words with apostrophes. RTF is summed decode time
over summed audio duration; model loading is excluded. Each talk and
condition starts a fresh process, so timing includes first-call overhead
and is diagnostic rather than a warmed serving benchmark. Eight talks do
not establish significance across speakers or recording types.

`--segmentation silence-split --windows 20,15,10` supplies the hard-split
baseline. Before each duration cap it searches the preceding 3 s on a
20 ms grid for the quietest 200 ms mean-square audio region, cutting at
its center. It retains every sample exactly once, without overlap. All
pieces from a talk are encoded in one batch, independently searched,
and their transcripts concatenated in source order. This is a
silence-favoring energy heuristic, not a VAD or diarizer; when there is
no silence it still picks the quietest region. Saved cuts are sample
offsets. `split_seconds` includes preparing and writing the piece WAVs;
the CLI's `decode_seconds` excludes that preparation. The comparison
tests overlap plus joining before search together, not center averaging
in isolation. Run the splitter checks with the recipe's dependencies:
`uv run --with numpy --with pyarrow --with soundfile --with rapidfuzz python -m unittest discover -s repro -p test_encoder_windows.py`.

### Other claims

Add its downloads under `[artifacts.<name>]` with a URL and SHA-256, then a
`[claims.<name>]` table naming those artifacts, the recognizer's flags, the
test set, and one `[[claims.<name>.arms]]` per configuration. A test set is
a LibriSpeech-layout archive for now.

## Checking the parallel program

[check_parallel.py](check_parallel.py) checks `sherpa-onnx-offline-parallel`:
threads, separate search threads and sorting by length must not change what
it prints. It decodes the three recordings shipped inside the pinned English
zipformer CTC archive, repeated, with one unreadable file among them, and
compares each way of running the program with one thread: a row per input
(in list order when `--ordered-output=true` asks for it, each index once
when rows are printed as they finish), an error row for the unreadable file
only, and the same text
at one recording per batch. It does this for three recognizers, so both
transducer implementations are covered as well as CTC:

| Recognizer | Search configurations |
|---|---|
| zipformer CTC | greedy; prefix search, also with max alignment, a narrow beam and no score limit; with a small n-gram model the script writes; with a tiny character LSTM |
| zipformer transducer (small English) | greedy; upstream's beam search; prefix search, also with modified_beam_search's rules, a narrow beam and no score limit; with the n-gram model |
| NeMo Parakeet TDT (110M English) | greedy; upstream's beam search; prefix search, also with a narrow beam and no score limit |

Each recognizer also runs greedy search and the n-gram search with the
encoder in 4 s windows ([chunked encoding](../topics/chunked-encoding.md)):
blended by center weight, cut, normalized per utterance, and with uniform
weights and context-only edges, and for CTC with probabilities averaged.

Where an option must change the output of the shipped recordings, the
check also requires that at one thread: the character LSTM and the window
settings on CTC and on the TDT model, and the narrow beam on the TDT model.
modified_beam_search's rules must give upstream's beam search's texts
exactly.

```bash
mise run check                                # against ../build
uv run check_parallel.py --build-dir DIR      # any build directory
uv run check_parallel.py --offline            # skip unless the model is fetched
uv run check_parallel.py --lstm-model FILE    # also an onnx-rnn token scorer
```

- A build without KenLM skips the n-gram configuration and says so.
- The LSTM is a 1.6 MB test fixture
  ([graehl/sherpa-onnx-fusion-test-lm](https://huggingface.co/graehl/sherpa-onnx-fusion-test-lm)),
  trained for 15 minutes so the check has a recurrent scorer to run. No
  claim may rest on its quality. `--lstm-model` adds a second, real one
  over the recognizer's tokens if you have it.
- It downloads about 600 MB of pinned public archives the first time, then
  takes a few minutes on an idle CPU, and needs no GPU. The workflow
  `.github/workflows/fusion-fork.yaml` runs it after the fork's unit tests.
- A developer's checkout can go further than the workflow does: add
  `--lstm-model`, or point `--build-dir` at a GPU build.

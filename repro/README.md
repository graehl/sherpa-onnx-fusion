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
- **What is not here.** The public token-LSTM CTC and transducer comparisons
  have no recipes yet. Claims that need unpublished language models also
  have no entries.

## Adding a claim

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
in list order, an error row for the unreadable file only, and the same text
at one recording per batch. It does this for three recognizers, so both
transducer implementations are covered as well as CTC:

| Recognizer | Search configurations |
|---|---|
| zipformer CTC | greedy; prefix search; with a small n-gram model the script writes; with a tiny character LSTM |
| zipformer transducer (small English) | greedy; upstream's beam search; prefix search; with the n-gram model |
| NeMo Parakeet TDT (110M English) | greedy; upstream's beam search; prefix search |

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

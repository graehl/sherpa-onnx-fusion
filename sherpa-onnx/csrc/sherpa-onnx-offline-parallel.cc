// sherpa-onnx/csrc/sherpa-onnx-offline-parallel.cc
//
// Copyright (c)  2022-2023  cuidc

#include <stdio.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/wave-reader.h"

std::atomic<int> wav_index(0);
std::mutex mtx;

std::vector<std::vector<std::string>> SplitToBatches(
    const std::vector<std::string> &input, int32_t batch_size) {
  std::vector<std::vector<std::string>> outputs;
  auto itr = input.cbegin();
  int32_t process_num = 0;

  while (process_num + batch_size <= static_cast<int32_t>(input.size())) {
    auto chunk_end = itr + batch_size;
    outputs.emplace_back(itr, chunk_end);
    itr = chunk_end;
    process_num += batch_size;
  }
  if (itr != input.cend()) {
    outputs.emplace_back(itr, input.cend());
  }
  return outputs;
}

std::vector<std::string> LoadScpFile(const std::string &wav_scp_path) {
  std::vector<std::string> wav_paths;
  auto in = sherpa_onnx::OpenInputFile(wav_scp_path);
  if (!in.is_open()) {
    fprintf(stderr, "Failed to open file: %s.\n", wav_scp_path.c_str());
    return wav_paths;
  }
  std::string line;
  while (std::getline(in, line)) {
    // Either "wav-id wav-path" or only "wav-path"
    std::string column1, column2;
    std::istringstream iss(line);
    iss >> column1 >> column2;
    if (column1.empty()) {
      continue;
    }
    wav_paths.emplace_back(column2.empty() ? std::move(column1)
                                           : std::move(column2));
  }

  return wav_paths;
}

// The result of one wave file
struct Record {
  std::string wav_filename;
  std::string json;
};

// Prints the results of each batch in the order of the input list, no matter
// which thread finishes first.
class OrderedOutput {
 public:
  void Put(int32_t chunk, std::vector<Record> records) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.emplace(chunk, std::move(records));
    for (auto it = pending_.begin();
         it != pending_.end() && it->first == next_chunk_;
         it = pending_.erase(it), ++next_chunk_) {
      for (const auto &r : it->second) {
        fprintf(stderr, "%s\n", r.wav_filename.c_str());
        fprintf(stdout, "%s\n", r.json.c_str());
        fprintf(stderr, "----\n");
      }
    }
    fflush(stdout);
  }

 private:
  std::mutex mutex_;
  std::map<int32_t, std::vector<Record>> pending_;  // finished, not printed
  int32_t next_chunk_ = 0;
};

// Decodes the given wave files as one batch.
//
// @param first_index Position of wav_paths[0] in the input list.
// @param duration On return, increased by the seconds of audio decoded.
// @return One record for each wave file, in the order of wav_paths. Each
//         json is a single line that starts with the position of the file
//         in the input list and its filename.
std::vector<Record> DecodeBatch(sherpa_onnx::OfflineRecognizer *recognizer,
                                const std::vector<std::string> &wav_paths,
                                int32_t first_index, float *duration) {
  // ss[i] is nullptr if wav_paths[i] cannot be read
  std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
  std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
  for (const auto &wav_filename : wav_paths) {
    int32_t sampling_rate = -1;
    bool is_ok = false;
    const std::vector<float> samples =
        sherpa_onnx::ReadWave(wav_filename, &sampling_rate, &is_ok);
    if (!is_ok) {
      fprintf(stderr, "Failed to read '%s'\n", wav_filename.c_str());
      ss.push_back(nullptr);
      continue;
    }
    *duration += samples.size() / static_cast<float>(sampling_rate);
    auto s = recognizer->CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());

    ss.push_back(std::move(s));
    ss_pointers.push_back(ss.back().get());
  }
  if (!ss_pointers.empty()) {
    recognizer->DecodeStreams(ss_pointers.data(), ss_pointers.size());
  }

  std::vector<Record> records;
  for (size_t i = 0; i != wav_paths.size(); ++i) {
    std::string json =
        "{\"index\": " + std::to_string(first_index + i) +
        ", \"wav_filename\": " + sherpa_onnx::QuoteJsonString(wav_paths[i]) +
        ", ";
    if (ss[i]) {
      // skip the leading '{' of the result
      json += ss[i]->GetResult().AsJsonString().substr(1);
    } else {
      json += "\"error\": \"failed to read the wave file\"}";
    }
    records.push_back({wav_paths[i], std::move(json)});
  }
  return records;
}

void AsrInference(const std::vector<std::vector<std::string>> &chunk_wav_paths,
                  int32_t batch_size, int32_t warm_up,
                  sherpa_onnx::OfflineRecognizer *recognizer,
                  OrderedOutput *output, float *total_length,
                  float *total_time) {
  float duration = 0.0f;
  float elapsed_seconds_batch = 0.0f;

  for (int32_t i = 0; i < warm_up; ++i) {
    float unused = 0.0f;
    DecodeBatch(recognizer, chunk_wav_paths[0], 0, &unused);
  }

  while (true) {
    int chunk = wav_index.fetch_add(1);
    if (chunk >= static_cast<int32_t>(chunk_wav_paths.size())) {
      break;
    }
    const auto &wav_paths = chunk_wav_paths[chunk];
    const auto begin = std::chrono::steady_clock::now();
    // every chunk before the last one has batch_size wave files
    std::vector<Record> records =
        DecodeBatch(recognizer, wav_paths, chunk * batch_size, &duration);
    const auto end = std::chrono::steady_clock::now();
    float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;
    elapsed_seconds_batch += elapsed_seconds;
    output->Put(chunk, std::move(records));
  }

  {
    std::lock_guard<std::mutex> guard(mtx);
    *total_length += duration;
    if (*total_time < elapsed_seconds_batch) {
      *total_time = elapsed_seconds_batch;
    }
  }
}

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Speech recognition using non-streaming models with sherpa-onnx.

Usage:

(1) Transducer from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

  ./bin/sherpa-onnx-offline-parallel \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    --batch-size=8 \
    --nj=1 \
    --wav-scp=wav.scp

  ./bin/sherpa-onnx-offline-parallel \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    --batch-size=1 \
    --nj=8 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(2) Paraformer from FunASR

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

  ./bin/sherpa-onnx-offline-parallel \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(3) Whisper models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html

  ./bin/sherpa-onnx-offline-parallel \
    --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(4) NeMo CTC models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

  ./bin/sherpa-onnx-offline-parallel \
    --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

(5) TDNN CTC model for the yesno recipe from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html
      //
  ./bin/sherpa-onnx-offline-parallel \
    --sample-rate=8000 \
    --feat-dim=23 \
    --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
    --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav

Note: It supports decoding multiple files in batches

Each result is one line of JSON on stdout. It starts with "index", the
position of the wave file in the input list counting from 0, and
"wav_filename". Results are printed in the order of the input list for any
--nj. A wave file that cannot be read has "error" in place of the result.

foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";
  std::string wav_scp = "";  // file path, kaldi style wav list.
  int32_t nj = 1;            // thread number
  int32_t batch_size = 1;    // number of wav files processed at once.
  int32_t warm_up = 1;       // untimed decodes of the first batch per thread
  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineRecognizerConfig config;
  config.Register(&po);
  po.Register("wav-scp", &wav_scp,
              "a file including wav-id and wav-path, kaldi style wav list."
              " A line may also contain only wav-path. "
              "default="
              ". when it is not empty, wav files which positional "
              "parameters provide are invalid.");
  po.Register("nj", &nj, "multi-thread num for decoding, default=1");
  po.Register("warm-up", &warm_up,
              "Number of times each thread decodes the first batch before "
              "the timed decoding starts. Its results are discarded. "
              "default=1");
  po.Register("batch-size", &batch_size,
              "number of wav files processed at once during the decoding"
              "process. default=1");

  po.Read(argc, argv);
  if (po.NumArgs() < 1 && wav_scp.empty()) {
    fprintf(stderr, "Error: Please provide at least 1 wave file.\n\n");
    po.PrintUsage();
    SHERPA_ONNX_EXIT(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }
  std::this_thread::sleep_for(std::chrono::seconds(10));  // sleep 10s
  fprintf(stderr, "Creating recognizer ...\n");
  const auto begin = std::chrono::steady_clock::now();
  sherpa_onnx::OfflineRecognizer recognizer(config);
  const auto end = std::chrono::steady_clock::now();
  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;
  fprintf(stderr,
          "Started nj: %d, batch_size: %d, wav_path: %s. recognizer init time: "
          "%.6f\n",
          nj, batch_size, wav_scp.c_str(), elapsed_seconds);
  std::this_thread::sleep_for(std::chrono::seconds(10));  // sleep 10s
  std::vector<std::string> wav_paths;
  if (!wav_scp.empty()) {
    wav_paths = LoadScpFile(wav_scp);
  } else {
    for (int32_t i = 1; i <= po.NumArgs(); ++i) {
      wav_paths.emplace_back(po.GetArg(i));
    }
  }
  if (wav_paths.empty()) {
    fprintf(stderr, "wav files is empty.\n");
    return -1;
  }
  std::vector<std::thread> threads;
  std::vector<std::vector<std::string>> batch_wav_paths =
      SplitToBatches(wav_paths, batch_size);
  float total_length = 0.0f;
  float total_time = 0.0f;
  OrderedOutput output;
  for (int i = 0; i < nj; i++) {
    threads.emplace_back(std::thread(AsrInference, std::cref(batch_wav_paths),
                                     batch_size, warm_up, &recognizer, &output,
                                     &total_length, &total_time));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
  if (config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
  }
  fprintf(stderr, "Elapsed seconds: %.3f s\n", total_time);
  float rtf = total_time / total_length;
  fprintf(stderr, "Real time factor (RTF): %.6f / %.6f = %.4f\n", total_time,
          total_length, rtf);
  fprintf(stderr, "SPEEDUP: %.4f\n", 1.0 / rtf);

  return 0;
}

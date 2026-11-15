// sherpa-onnx/csrc/sherpa-onnx-offline-parallel.cc
//
// Copyright (c)  2022-2023  cuidc

#include <stdio.h>
#include <string.h>

#include <algorithm>
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
#include <tuple>
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

// A wave file to decode
struct Utterance {
  int32_t index;  // position in the input list
  std::string wav_filename;
  float seconds;  // estimated duration; 0 if unknown
};

// Estimates the duration of a wave file in seconds from its size and the byte
// rate in its header, without reading its samples. Returns 0 if the file
// cannot be read or has no usable byte rate.
float EstimateSeconds(const std::string &filename) {
  auto in = sherpa_onnx::OpenInputFile(filename, std::ios::binary);
  char header[44];  // the size of a wave header without extra chunks
  if (!in.is_open() || !in.read(header, sizeof(header))) {
    return 0;
  }
  int32_t byte_rate = 0;
  memcpy(&byte_rate, header + 28, sizeof(byte_rate));
  in.seekg(0, std::ios::end);
  int64_t num_bytes = static_cast<int64_t>(in.tellg()) - sizeof(header);
  return byte_rate > 0 ? num_bytes / static_cast<float>(byte_rate) : 0;
}

// Cuts the input into consecutive batches of batch_size wave files. The last
// batch has the remaining ones.
std::vector<std::vector<Utterance>> SplitToBatches(
    const std::vector<Utterance> &input, int32_t batch_size) {
  std::vector<std::vector<Utterance>> outputs;
  for (const auto &u : input) {
    if (outputs.empty() ||
        static_cast<int32_t>(outputs.back().size()) >= batch_size) {
      outputs.emplace_back();
    }
    outputs.back().push_back(u);
  }
  return outputs;
}

// What a way of cutting some wave files into batches costs. The smaller the
// better, comparing the fields in the order they are declared.
struct BatchingCost {
  // Modelled time of the batches, in milliseconds of audio. A batch takes a
  // fixed overhead plus the audio it holds after every wave file is padded
  // to the longest of the batch, that is, its longest duration times its
  // number of wave files.
  int64_t milliseconds = 0;
  // Sum of the squared numbers of wave files of the batches. For a given
  // number of batches it is smallest when the batches are equally full.
  int64_t sum_squared_counts = 0;

  bool operator<(const BatchingCost &other) const {
    return std::tie(milliseconds, sum_squared_counts) <
           std::tie(other.milliseconds, other.sum_squared_counts);
  }
};

// Cuts the input into consecutive batches of the least BatchingCost. An
// extra batch is worth its overhead only if it saves more padding than that,
// and among equally good ways the wave files are spread evenly, which
// leaves no nearly empty batch.
//
// @param batch_size The maximum number of wave files in a batch.
// @param max_batch_seconds If positive, the longest estimated duration in a
//        batch times its number of wave files is at most this. A wave file
//        that exceeds it alone is a batch by itself.
// @param batch_overhead_seconds The fixed time of one call to the recognizer,
//        as the seconds of audio the model processes in that time.
std::vector<std::vector<Utterance>> SplitToSimilarBatches(
    const std::vector<Utterance> &input, int32_t batch_size,
    float max_batch_seconds, float batch_overhead_seconds) {
  int32_t n = input.size();
  int64_t max_milliseconds = max_batch_seconds * 1000;
  int64_t overhead_milliseconds = batch_overhead_seconds * 1000;
  // best[i] is the least cost of cutting the first i wave files, and its
  // last batch starts at input[start[i]]
  std::vector<BatchingCost> best(n + 1);
  std::vector<int32_t> start(n + 1, 0);
  for (int32_t i = 1; i <= n; ++i) {
    int64_t longest = 0;  // milliseconds, in input[j] to input[i - 1]
    for (int32_t j = i - 1; j >= 0 && (i - j <= batch_size || j == i - 1);
         --j) {
      longest =
          std::max(longest, static_cast<int64_t>(input[j].seconds * 1000));
      int64_t count = i - j;
      if (max_milliseconds > 0 && count > 1 &&
          longest * count > max_milliseconds) {
        break;  // an even earlier start holds no less
      }
      BatchingCost cost = best[j];
      cost.milliseconds += overhead_milliseconds + longest * count;
      cost.sum_squared_counts += count * count;
      if (j == i - 1 || cost < best[i]) {
        best[i] = cost;
        start[i] = j;
      }
    }
  }

  std::vector<std::vector<Utterance>> outputs;
  for (int32_t i = n; i > 0; i = start[i]) {
    outputs.emplace_back(input.begin() + start[i], input.begin() + i);
  }
  std::reverse(outputs.begin(), outputs.end());
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
  int32_t index;  // position in the input list
  std::string wav_filename;
  std::string json;
};

// Prints the results in the order of the input list, no matter in which
// order they are decoded.
class OrderedOutput {
 public:
  void Put(std::vector<Record> records) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &r : records) {
      int32_t index = r.index;
      pending_.emplace(index, std::move(r));
    }
    for (auto it = pending_.begin();
         it != pending_.end() && it->first == next_index_;
         it = pending_.erase(it), ++next_index_) {
      const Record &r = it->second;
      fprintf(stderr, "%s\n", r.wav_filename.c_str());
      fprintf(stdout, "%s\n", r.json.c_str());
      fprintf(stderr, "----\n");
    }
    fflush(stdout);
  }

 private:
  std::mutex mutex_;
  std::map<int32_t, Record> pending_;  // finished, not printed
  int32_t next_index_ = 0;
};

// Decodes the given wave files as one batch.
//
// @param duration On return, increased by the seconds of audio decoded.
// @param recognizer_seconds On return, increased by the seconds spent in the
//        recognizer, which excludes reading files and computing features.
// @return One record for each wave file, in the order of utterances. Each
//         json is a single line that starts with the position of the file
//         in the input list and its filename.
std::vector<Record> DecodeBatch(sherpa_onnx::OfflineRecognizer *recognizer,
                                const std::vector<Utterance> &utterances,
                                float *duration, float *recognizer_seconds) {
  // ss[i] is nullptr if utterances[i] cannot be read
  std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
  std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
  for (const auto &utterance : utterances) {
    const std::string &wav_filename = utterance.wav_filename;
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
    const auto begin = std::chrono::steady_clock::now();
    recognizer->DecodeStreams(ss_pointers.data(), ss_pointers.size());
    *recognizer_seconds +=
        std::chrono::duration<float>(std::chrono::steady_clock::now() - begin)
            .count();
  }

  std::vector<Record> records;
  for (size_t i = 0; i != utterances.size(); ++i) {
    const Utterance &u = utterances[i];
    std::string json =
        "{\"index\": " + std::to_string(u.index) +
        ", \"wav_filename\": " + sherpa_onnx::QuoteJsonString(u.wav_filename) +
        ", ";
    if (ss[i]) {
      // skip the leading '{' of the result
      json += ss[i]->GetResult().AsJsonString().substr(1);
    } else {
      json += "\"error\": \"failed to read the wave file\"}";
    }
    records.push_back({u.index, u.wav_filename, std::move(json)});
  }
  return records;
}

void AsrInference(const std::vector<std::vector<Utterance>> &chunk_wav_paths,
                  int32_t warm_up, sherpa_onnx::OfflineRecognizer *recognizer,
                  OrderedOutput *output, float *total_length, float *total_time,
                  float *total_recognizer_time) {
  float duration = 0.0f;
  float elapsed_seconds_batch = 0.0f;
  float recognizer_seconds = 0.0f;

  for (int32_t i = 0; i < warm_up; ++i) {
    float unused = 0.0f;
    DecodeBatch(recognizer, chunk_wav_paths[0], &unused, &unused);
  }

  while (true) {
    int chunk = wav_index.fetch_add(1);
    if (chunk >= static_cast<int32_t>(chunk_wav_paths.size())) {
      break;
    }
    const auto &wav_paths = chunk_wav_paths[chunk];
    const auto begin = std::chrono::steady_clock::now();
    std::vector<Record> records =
        DecodeBatch(recognizer, wav_paths, &duration, &recognizer_seconds);
    const auto end = std::chrono::steady_clock::now();
    float elapsed_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count() /
        1000.;
    elapsed_seconds_batch += elapsed_seconds;
    output->Put(std::move(records));
  }

  {
    std::lock_guard<std::mutex> guard(mtx);
    *total_length += duration;
    if (*total_time < elapsed_seconds_batch) {
      *total_time = elapsed_seconds_batch;
    }
    *total_recognizer_time += recognizer_seconds;
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

With --sort-by-length=true a batch holds wave files of similar length, which
avoids computing on padding. --max-batch-seconds limits a batch by its
longest wave file times its number of wave files.

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
  bool sort_by_length = false;  // batch wave files of similar length
  float max_batch_seconds = 0;  // limit on a padded batch; 0 means none
  // fixed time of a recognizer call in seconds of audio; the default is so
  // large that the fewest possible batches are used
  float batch_overhead_seconds = 1e6;
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
  po.Register("sort-by-length", &sort_by_length,
              "If true, decode the wave files from the longest to the "
              "shortest so that a batch holds files of similar length and "
              "needs little padding. Results are still printed in the order "
              "of the input list. The length is estimated from the file "
              "size and the wave header. Note that for some models the "
              "result of a wave file depends on the others in its batch. "
              "default=false");
  po.Register("batch-overhead-seconds", &batch_overhead_seconds,
              "Used with --sort-by-length or --max-batch-seconds. The fixed "
              "time of one call to the recognizer, given as the seconds of "
              "audio the model processes in that time. Batches are chosen "
              "to minimize this overhead for every batch plus the audio "
              "the batches hold after padding, so a smaller value gives "
              "more batches with less padding. The default is so large "
              "that the fewest possible batches are used. default=1e6");
  po.Register("max-batch-seconds", &max_batch_seconds,
              "If positive, a batch is also limited so that the duration of "
              "its longest wave file times its number of wave files is at "
              "most this many seconds. A longer wave file is decoded alone. "
              "--batch-size still limits the number of wave files. "
              "default=0");

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
  std::vector<Utterance> utterances;
  for (size_t i = 0; i != wav_paths.size(); ++i) {
    float seconds = (sort_by_length || max_batch_seconds > 0)
                        ? EstimateSeconds(wav_paths[i])
                        : 0;
    utterances.push_back({static_cast<int32_t>(i), wav_paths[i], seconds});
  }
  if (sort_by_length) {
    // Longest first, so that the largest batch is decoded first
    std::stable_sort(utterances.begin(), utterances.end(),
                     [](const Utterance &a, const Utterance &b) {
                       return a.seconds > b.seconds;
                     });
  }
  std::vector<std::thread> threads;
  std::vector<std::vector<Utterance>> batch_wav_paths =
      (sort_by_length || max_batch_seconds > 0)
          ? SplitToSimilarBatches(utterances, batch_size, max_batch_seconds,
                                  batch_overhead_seconds)
          : SplitToBatches(utterances, batch_size);
  fprintf(stderr, "Number of wave files: %d, number of batches: %d\n",
          static_cast<int32_t>(utterances.size()),
          static_cast<int32_t>(batch_wav_paths.size()));
  if (sort_by_length || max_batch_seconds > 0) {
    float audio = 0;   // estimated seconds of all wave files
    float padded = 0;  // the same after padding to the longest of each batch
    size_t smallest = utterances.size();
    for (const auto &batch : batch_wav_paths) {
      float longest = 0;
      for (const auto &u : batch) {
        audio += u.seconds;
        longest = std::max(longest, u.seconds);
      }
      padded += longest * batch.size();
      smallest = std::min(smallest, batch.size());
    }
    fprintf(stderr,
            "Estimated audio: %.1f s, with padding: %.1f s, smallest batch: "
            "%d wave files\n",
            audio, padded, static_cast<int32_t>(smallest));
  }
  float total_length = 0.0f;
  float total_time = 0.0f;
  float total_recognizer_time = 0.0f;
  OrderedOutput output;
  for (int i = 0; i < nj; i++) {
    threads.emplace_back(std::thread(
        AsrInference, std::cref(batch_wav_paths), warm_up, &recognizer, &output,
        &total_length, &total_time, &total_recognizer_time));
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
  fprintf(stderr, "Seconds in the recognizer, summed over threads: %.3f s\n",
          total_recognizer_time);
  float rtf = total_time / total_length;
  fprintf(stderr, "Real time factor (RTF): %.6f / %.6f = %.4f\n", total_time,
          total_length, rtf);
  fprintf(stderr, "SPEEDUP: %.4f\n", 1.0 / rtf);

  return 0;
}

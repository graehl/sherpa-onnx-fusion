// sherpa-onnx/csrc/offline-encoder-window.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_ENCODER_WINDOW_H_
#define SHERPA_ONNX_CSRC_OFFLINE_ENCODER_WINDOW_H_

#include <cstdint>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

// How an offline encoder runs an utterance longer than a window: in
// overlapping windows whose outputs are blended per output frame
// (topics/chunked-encoding.md). Registered as --encoder.window.*; with
// seconds == 0, the default, every utterance is encoded whole.
struct OfflineEncoderWindowConfig {
  // Window length. 0: no windows. A LiteRT model of several fixed lengths
  // takes its shortest length of at least this, and 0 means its longest.
  float seconds = 0;
  // The least overlap of neighbouring windows.
  float overlap = 3;
  // How far an interior window may move to put its edges where the
  // filterbank energy is lowest; 0 keeps the windows evenly spaced.
  float slack = 1;
  // Scope of a normalization that uses statistics of its input:
  // "window" (each window as an utterance of its own) or "utterance".
  // Normalizations with fixed statistics are the same either way.
  std::string normalize = "window";
  // "center": a frame is the average of the windows that cover it, each
  // weighted by 1 + the frame's distance from that window's nearer edge;
  // "cut": a frame is taken from the window in which it is farthest from
  // an edge.
  std::string weight = "center";
  // For CTC outputs, what "center" averages: "log", log probabilities (or
  // logits), renormalized afterwards; "prob", probabilities. A
  // transducer's encoder output is averaged as it is.
  std::string blend = "log";

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;

  bool Enabled() const { return seconds > 0; }
};

// The options of running an offline model's encoder, registered under
// --encoder.
struct OfflineEncoderConfig {
  OfflineEncoderWindowConfig window;
  // The input lengths an encoder call may have, so that an accelerator sees
  // few distinct shapes: every input, window or whole utterance, is padded
  // to the shortest of `lengths` (seconds, comma separated) that holds it,
  // else to a multiple of `pad_to` seconds, else not at all, and its true
  // length is passed. A window is the shortest such length that holds
  // window.seconds. Both need a model that takes a padded batch.
  std::string lengths;
  float pad_to = 0;

  // `lengths` in seconds, sorted; false on a malformed list.
  bool ParseLengths(std::vector<float> *seconds) const;

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

// A window schedule in input units of the caller's choice (feature frames,
// or a LiteRT graph's input rows).
struct WindowSchedule {
  int32_t window = 0;
  int32_t min_overlap = 0;
  int32_t slack = 0;
};

// config's lengths in units of `unit_seconds`; window is 0 when off.
WindowSchedule ScheduleInUnits(const OfflineEncoderWindowConfig &config,
                               float unit_seconds);

// Window starts covering `length` units with windows of `window` units, each
// start a multiple of `grid` (the units per output frame): as few as keep
// neighbours overlapping by at least s.min_overlap, evenly spaced, the first
// at 0 and the last ending at or past the end; then each interior window
// moves by up to s.slack to the start whose two edge units are quietest
// (lowest `loudness`, one value per unit), keeping half the minimum overlap
// with both neighbours. min_coverage_overlap, in input units, is a hard
// floor retained through alignment and slack for encoders with edge loss.
// One start, 0, when length <= window.
std::vector<int32_t> EvenWindowStarts(int32_t length, int32_t window,
                                      const WindowSchedule &s,
                                      const std::vector<float> &loudness,
                                      int32_t grid,
                                      int32_t min_coverage_overlap = 0);

// Accumulates windows' output rows into one sequence of output frames.
class WindowBlend {
 public:
  // What the rows are: hidden states, averaged as they are; or a CTC
  // model's scores over its vocabulary, made log probabilities on Finish.
  enum class Rows { kHidden, kScores };

  // `config` gives weight and blend.
  WindowBlend(int32_t num_frames, int32_t dim, Rows rows,
              const OfflineEncoderWindowConfig &config);

  // Adds one window's `frames` output rows, the first at output frame
  // `offset`.
  void Add(const float *rows, int32_t frames, int32_t offset);

  // Writes num_frames x dim blended rows to `out`. Returns false, writing
  // nothing, if a frame is in no window.
  bool Finish(float *out) const;

 private:
  enum class Mode { kCenterLinear, kCenterProb, kCut };

  int32_t num_frames_;
  int32_t dim_;
  bool log_softmax_;
  Mode mode_;
  std::vector<double> sum_;     // per frame and value
  std::vector<double> weight_;  // per frame
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_ENCODER_WINDOW_H_

// sherpa-onnx/csrc/offline-encoder-window.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-encoder-window.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

void OfflineEncoderWindowConfig::Register(ParseOptions *po) {
  ParseOptions p("window", po);
  p.Register("seconds", &seconds,
             "Encode an utterance longer than this many seconds in "
             "overlapping windows of this length, blended per output frame. "
             "0: encode every utterance whole");
  p.Register("overlap", &overlap,
             "Least overlap of neighbouring encoder windows, in seconds");
  p.Register("slack", &slack,
             "Seconds an interior encoder window may move to put its edges "
             "where the audio is quietest; 0: evenly spaced windows");
  p.Register("normalize", &normalize,
             "window or utterance: the frames over which a feature "
             "normalization that uses its input's statistics is computed");
  p.Register("weight", &weight,
             "center: average the windows covering a frame, weighted by "
             "distance from each window's nearer edge; cut: take the frame "
             "from the window in which it is farthest from an edge");
  p.Register("blend", &blend,
             "log or prob: for a CTC model, average log probabilities "
             "and renormalize, or average probabilities");
  p.Register("center-power", &center_power,
             "Center-weight exponent: 0 uniform, 1 linear, larger sharper; "
             "finite and nonnegative; requires weight=center");
  p.Register("context-frames", &context_frames,
             "Zero-weight output frames per interior window edge; recording "
             "ends retained; exclusive with context-fraction");
  p.Register("context-fraction", &context_fraction,
             "Zero-weight fraction per interior window edge, rounded down "
             "to output frames; [0,0.5); exclusive with context-frames");
}

bool OfflineEncoderWindowConfig::Validate() const {
  if (!std::isfinite(center_power) || center_power < 0 || context_frames < 0 ||
      !std::isfinite(context_fraction) || context_fraction < 0 ||
      context_fraction >= 0.5f ||
      (context_frames > 0 && context_fraction > 0)) {
    SHERPA_ONNX_LOGE(
        "encoder.window: center-power must be finite and nonnegative; "
        "context-frames nonnegative; context-fraction in [0,0.5); "
        "choose only one context setting");
    return false;
  }
  if (weight != "center" &&
      (center_power != 1 || context_frames != 0 || context_fraction != 0)) {
    SHERPA_ONNX_LOGE(
        "encoder.window center-power and context settings require "
        "weight=center");
    return false;
  }
  if (!std::isfinite(seconds) || !std::isfinite(overlap) ||
      !std::isfinite(slack) || seconds < 0 || overlap < 0 || slack < 0) {
    SHERPA_ONNX_LOGE(
        "--encoder.window.seconds, overlap and slack must be finite and "
        "nonnegative. Given %g, %g, %g",
        seconds, overlap, slack);
    return false;
  }
  if (seconds > 0 && overlap >= seconds) {
    SHERPA_ONNX_LOGE(
        "--encoder.window.overlap (%g) must be shorter than "
        "--encoder.window.seconds (%g)",
        overlap, seconds);
    return false;
  }
  if (normalize != "window" && normalize != "utterance") {
    SHERPA_ONNX_LOGE(
        "--encoder.window.normalize is window or utterance. Given %s",
        normalize.c_str());
    return false;
  }
  if (weight != "center" && weight != "cut") {
    SHERPA_ONNX_LOGE("--encoder.window.weight is center or cut. Given %s",
                     weight.c_str());
    return false;
  }
  if (blend != "log" && blend != "prob") {
    SHERPA_ONNX_LOGE("--encoder.window.blend is log or prob. Given %s",
                     blend.c_str());
    return false;
  }
  return true;
}

std::string OfflineEncoderWindowConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineEncoderWindowConfig(seconds=" << seconds
     << ", overlap=" << overlap << ", slack=" << slack << ", normalize=\""
     << normalize << "\", weight=\"" << weight << "\", blend=\"" << blend
     << "\", center_power=" << center_power
     << ", context_frames=" << context_frames
     << ", context_fraction=" << context_fraction << ")";
  return os.str();
}

void OfflineEncoderConfig::Register(ParseOptions *po) {
  ParseOptions p("encoder", po);
  window.Register(&p);
  p.Register("lengths", &lengths,
             "Comma-separated seconds: pad every encoder input, window or "
             "whole utterance, to the shortest of these that holds it, so "
             "that a GPU or NPU sees few distinct lengths; needs a model that "
             "masks padding. Empty: none");
  p.Register("pad-to", &pad_to,
             "Pad an encoder input that --encoder.lengths does not hold to a "
             "multiple of this many seconds; needs a model that masks "
             "padding. 0: none");
}

bool OfflineEncoderConfig::ParseLengths(std::vector<float> *seconds) const {
  seconds->clear();
  if (lengths.empty()) return true;
  std::istringstream in(lengths);
  for (std::string part; std::getline(in, part, ',');) {
    std::istringstream number(part);
    float value = 0;
    if (!(number >> value) || !(number >> std::ws).eof() ||
        !std::isfinite(value) || value <= 0) {
      return false;
    }
    seconds->push_back(value);
  }
  if (lengths.back() == ',') return false;
  std::sort(seconds->begin(), seconds->end());
  return true;
}

bool OfflineEncoderConfig::Validate() const {
  if (!std::isfinite(pad_to) || pad_to < 0) {
    SHERPA_ONNX_LOGE("--encoder.pad-to must be nonnegative. Given %g", pad_to);
    return false;
  }
  std::vector<float> seconds;
  if (!ParseLengths(&seconds)) {
    SHERPA_ONNX_LOGE(
        "--encoder.lengths is a comma-separated list of positive seconds. "
        "Given %s",
        lengths.c_str());
    return false;
  }
  return window.Validate();
}

std::string OfflineEncoderConfig::ToString() const {
  std::ostringstream os;
  os << "OfflineEncoderConfig(window=" << window.ToString() << ", lengths=\""
     << lengths << "\", pad_to=" << pad_to << ")";
  return os.str();
}

WindowSchedule ScheduleInUnits(const OfflineEncoderWindowConfig &config,
                               float unit_seconds) {
  auto units = [unit_seconds](float seconds) {
    double u = std::round(seconds / unit_seconds);
    if (u > std::numeric_limits<int32_t>::max() / 2) {
      SHERPA_ONNX_LOGE("An --encoder.window length is too long: %g s", seconds);
      SHERPA_ONNX_EXIT(-1);
    }
    return static_cast<int32_t>(u);
  };
  WindowSchedule s;
  s.window = units(config.seconds);
  s.min_overlap = units(config.overlap);
  s.slack = units(config.slack);
  return s;
}

std::vector<int32_t> EvenWindowStarts(int32_t length, int32_t window,
                                      const WindowSchedule &s,
                                      const std::vector<float> &loudness,
                                      int32_t grid,
                                      int32_t min_coverage_overlap) {
  if (length <= window) return {0};
  const int32_t step = window - s.min_overlap;
  if (step < grid) {
    SHERPA_ONNX_LOGE(
        "The encoder window overlap leaves no step: window %d, overlap %d, "
        "output frame %d input units",
        window, s.min_overlap, grid);
    SHERPA_ONNX_EXIT(-1);
  }
  int32_t count = 1 + (length - window + step - 1) / step;
  int32_t last = length - window;
  last += (grid - last % grid) % grid;
  if (min_coverage_overlap > 0) {
    const int32_t covered_step = (window - min_coverage_overlap) / grid * grid;
    if (covered_step < grid) {
      SHERPA_ONNX_LOGE("Encoder window is too short for its edge loss");
      SHERPA_ONNX_EXIT(-1);
    }
    count = std::max(count, 1 + (last + covered_step - 1) / covered_step);
  }
  std::vector<int32_t> starts(count);
  for (int32_t i = 0; i < count; ++i) {
    starts[i] = count == 1 ? 0
                           : static_cast<int32_t>(static_cast<int64_t>(last) *
                                                  i / (count - 1));
    starts[i] -= starts[i] % grid;
  }
  starts.back() = last;
  auto edges = [&](int32_t start) {
    return loudness[start] + loudness[std::min(start + window, length) - 1];
  };
  const int32_t keep = std::max(s.min_overlap / 2, min_coverage_overlap);
  for (int32_t i = 1; i + 1 < count; ++i) {
    int32_t lo = std::max({starts[i] - s.slack, starts[i - 1] + grid,
                           starts[i + 1] - window + keep});
    int32_t hi = std::min({starts[i] + s.slack, starts[i + 1] - grid,
                           starts[i - 1] + window - keep});
    lo += (grid - lo % grid) % grid;
    int32_t chosen = starts[i];
    for (int32_t start = lo; start <= hi; start += grid) {
      if (edges(start) < edges(chosen)) chosen = start;
    }
    starts[i] = chosen;
  }
  return starts;
}

WindowBlend::WindowBlend(int32_t num_frames, int32_t dim, Rows rows,
                         const OfflineEncoderWindowConfig &config)
    : num_frames_(num_frames),
      dim_(dim),
      log_softmax_(rows == Rows::kScores),
      mode_(config.weight == "cut" ? Mode::kCut
            : rows == Rows::kScores && config.blend == "prob"
                ? Mode::kCenterProb
                : Mode::kCenterLinear),
      center_power_(config.center_power),
      context_frames_(config.context_frames),
      context_fraction_(config.context_fraction),
      sum_(static_cast<size_t>(num_frames) * dim,
           mode_ == Mode::kCenterProb ? -std::numeric_limits<double>::infinity()
                                      : 0.0),
      weight_(num_frames, 0.0),
      log_scale_(
          config.center_power == 0 || config.center_power == 1 ? 0 : num_frames,
          -std::numeric_limits<double>::infinity()) {
  // An average of probabilities is already normalized.
  if (mode_ == Mode::kCenterProb) log_softmax_ = false;
}

void WindowBlend::Add(const float *rows, int32_t frames, int32_t offset) {
  const int32_t context = std::max(
      context_frames_,
      static_cast<int32_t>(frames * static_cast<double>(context_fraction_)));
  const int32_t left = offset == 0 ? 0 : context;
  const int32_t right = offset + frames == num_frames_ ? 0 : context;
  for (int32_t f = 0; f < frames; ++f, rows += dim_) {
    if (f < left || frames - 1 - f < right) continue;
    double w = 1.0 + std::min(f - left, frames - right - 1 - f);
    double *to = sum_.data() + static_cast<size_t>(offset + f) * dim_;
    double &total = weight_[offset + f];
    if (center_power_ == 0) {
      w = 1;
    } else if (center_power_ != 1) {
      const double log_w = center_power_ * std::log(w);
      double &scale = log_scale_[offset + f];
      if (log_w > scale) {
        const double shift = scale - log_w;
        const double factor = std::exp(shift);
        for (int32_t v = 0; v < dim_; ++v) {
          if (mode_ == Mode::kCenterProb) {
            to[v] += shift;
          } else {
            to[v] *= factor;
          }
        }
        total *= factor;
        scale = log_w;
      }
      w = std::exp(log_w - scale);
      if (w == 0) continue;
    }
    switch (mode_) {
      case Mode::kCenterLinear:
        for (int32_t v = 0; v < dim_; ++v) to[v] += w * rows[v];
        total += w;
        break;
      case Mode::kCenterProb: {
        double top = *std::max_element(rows, rows + dim_);
        double sum = 0;
        for (int32_t v = 0; v < dim_; ++v) sum += std::exp(rows[v] - top);
        // log w + log p(v), added into the log of the weighted sum.
        const double shift = std::log(w) - top - std::log(sum);
        for (int32_t v = 0; v < dim_; ++v) {
          double x = rows[v] + shift;
          double hi = std::max(to[v], x);
          if (hi == -std::numeric_limits<double>::infinity()) continue;
          to[v] = hi + std::log1p(std::exp(std::min(to[v], x) - hi));
        }
        total += w;
        break;
      }
      case Mode::kCut:
        if (w > total) {
          std::copy(rows, rows + dim_, to);
          total = w;
        }
        break;
    }
  }
}

bool WindowBlend::Finish(float *out) const {
  for (double w : weight_) {
    if (w <= 0) return false;
  }
  for (int32_t f = 0; f < num_frames_; ++f) {
    const double *from = sum_.data() + static_cast<size_t>(f) * dim_;
    float *to = out + static_cast<size_t>(f) * dim_;
    const double w = weight_[f];
    for (int32_t v = 0; v < dim_; ++v) {
      switch (mode_) {
        case Mode::kCenterLinear:
          to[v] = static_cast<float>(from[v] / w);
          break;
        case Mode::kCenterProb:
          to[v] = static_cast<float>(from[v] - std::log(w));
          break;
        case Mode::kCut:
          to[v] = static_cast<float>(from[v]);
          break;
      }
    }
  }
  if (log_softmax_) LogSoftmax(out, dim_, num_frames_);
  return true;
}

}  // namespace sherpa_onnx

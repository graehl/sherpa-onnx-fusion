// sherpa-onnx/csrc/transducer-survival-summary.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/transducer-survival-summary.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

using Histogram = TransducerWaitHistogram;

std::string Format(const char *format, double a, double b = 0) {
  char text[64];
  snprintf(text, sizeof(text), format, a, b);
  return text;
}

std::string RankLabel(int32_t r) {
  return r + 1 == Histogram::kNumRanks ? "any"
                                       : std::to_string(Histogram::kRanks[r]);
}

std::string GapLabel(int32_t g) {
  return g + 1 == Histogram::kNumGaps ? "any"
                                      : Format("%g", Histogram::kGaps[g]);
}

// The q quantile of "values", 0 for none.
template <typename T>
T Quantile(std::vector<T> values, double q) {
  if (values.empty()) return T();
  std::sort(values.begin(), values.end());
  size_t i = static_cast<size_t>(std::ceil(q * values.size()));
  return values[std::min(values.size() - 1, i == 0 ? 0 : i - 1)];
}

// One histogram of length offsets as percentages, from -kMaxOffset.
std::string Lengths(const std::vector<int64_t> &counts) {
  int64_t total = 0;
  for (int64_t n : counts) total += n;
  std::string out;
  for (size_t i = 0; i != counts.size(); ++i) {
    out += Format(" %5.1f", total ? 100.0 * counts[i] / total : 0.0);
  }
  return out + Format("  (%.0f)", static_cast<double>(total));
}

}  // namespace

TransducerSurvivalSummary::~TransducerSurvivalSummary() {
  if (searches_ != 0) SHERPA_ONNX_LOGE("%s", Report().c_str());
}

void TransducerSurvivalSummary::Record(const TransducerPrefixSearch &search,
                                       const TransducerPrefixHypothesis *best) {
  std::lock_guard<std::mutex> lock(mutex_);
  waiting_.Merge(search.WaitHistogram());
  ++searches_;
  if (best == nullptr) return;
  needs_.push_back(best->required);
  dominant_needs_.push_back(best->required_dominant);
}

namespace {

// Median, 95th percentile and largest of what "needs" asked of the search's
// own pruning.
std::string UseQuantiles(const std::vector<TransducerSurvivalNeed> &needs) {
  std::vector<int32_t> ranks;
  std::vector<float> gaps;
  for (const auto &need : needs) {
    ranks.push_back(need.rank);
    gaps.push_back(need.gap);
  }
  return "max-active-paths " + std::to_string(Quantile(ranks, 0.5)) + " / " +
         std::to_string(Quantile(ranks, 0.95)) + " / " +
         std::to_string(Quantile(ranks, 1.0)) + ", score-beam " +
         Format("%.2f", Quantile(gaps, 0.5)) + " / " +
         Format("%.2f", Quantile(gaps, 0.95)) + " / " +
         Format("%.2f", Quantile(gaps, 1.0));
}

// The share of "needs" that a cut to rank r and gap g of the histogram's
// bounds keeps.
double Unchanged(const std::vector<TransducerSurvivalNeed> &needs, int32_t r,
                 int32_t g) {
  if (needs.empty()) return 100.0;
  int32_t unchanged = 0;
  for (const auto &need : needs) {
    bool rank_ok =
        r + 1 == Histogram::kNumRanks || need.wait_rank <= Histogram::kRanks[r];
    bool gap_ok =
        g + 1 == Histogram::kNumGaps || need.wait_gap <= Histogram::kGaps[g];
    unchanged += rank_ok && gap_ok;
  }
  return 100.0 * unchanged / needs.size();
}

}  // namespace

std::string TransducerSurvivalSummary::Report() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string out =
      "\nTransducer survival trace: " + std::to_string(searches_) +
      " searches, " + std::to_string(needs_.size()) + " with a result, " +
      std::to_string(waiting_.rounds) + " round ends, at most " +
      std::to_string(waiting_.peak_waiting) + " entries waiting at once.\n";
  out +=
      "What best results needed of pruning when a queue is used (median / "
      "95% / most):\n  every path:    " +
      UseQuantiles(needs_) +
      "\n  dominant path: " + UseQuantiles(dominant_needs_) + "\n";

  // Cumulative counts: entries a cut to ranks[r] within gaps[g] keeps.
  const int32_t R = Histogram::kNumRanks, G = Histogram::kNumGaps;
  std::vector<double> kept(static_cast<size_t>(R) * G, 0);
  for (int32_t r = 0; r != R; ++r) {
    for (int32_t g = 0; g != G; ++g) {
      double v = static_cast<double>(waiting_.counts[r * G + g]);
      if (r > 0) v += kept[(r - 1) * G + g];
      if (g > 0) v += kept[r * G + g - 1];
      if (r > 0 && g > 0) v -= kept[(r - 1) * G + g - 1];
      kept[r * G + g] = v;
    }
  }
  const double rounds = std::max<double>(waiting_.rounds, 1);
  out +=
      "Cutting each waiting list at every round end to its best K within B "
      "of its best. Each cell: percent of best results whose every path "
      "survives, percent whose dominant path does (to first order), and "
      "the mean entries kept per round end (" +
      Format("%.1f", kept.back() / rounds) + " now).\n";
  // Each cell is "  ddd.d ddd.d eeeee": 19 characters under a right-aligned B.
  auto pad = [](const std::string &text, size_t width) {
    return std::string(width - std::min(width, text.size()), ' ') + text;
  };
  out += pad("K \\ B", 6);
  for (int32_t g = 0; g != G; ++g) out += pad(GapLabel(g), 19);
  out += "\n";
  for (int32_t r = 0; r != R; ++r) {
    out += pad(RankLabel(r), 6);
    for (int32_t g = 0; g != G; ++g) {
      out += Format("  %5.1f", Unchanged(needs_, r, g)) +
             Format(" %5.1f", Unchanged(dominant_needs_, r, g)) +
             Format(" %5.0f", kept[r * G + g] / rounds);
    }
    out += "\n";
  }
  out += "Prefix length minus that of the list's best, -" +
         std::to_string(Histogram::kMaxOffset) + " to +" +
         std::to_string(Histogram::kMaxOffset) +
         " (ends clamped), percent (total):\n";
  out += "  waiting at round ends:" + Lengths(waiting_.waiting_lengths) + "\n";
  out += "  kept on arrival:      " + Lengths(waiting_.kept_lengths) + "\n";
  out += "  dropped on arrival:   " + Lengths(waiting_.dropped_lengths) + "\n";
  return out;
}

}  // namespace sherpa_onnx

// sherpa-onnx/csrc/transducer-survival-summary.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_TRANSDUCER_SURVIVAL_SUMMARY_H_
#define SHERPA_ONNX_CSRC_TRANSDUCER_SURVIVAL_SUMMARY_H_

#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/transducer-prefix-search.h"

namespace sherpa_onnx {

// Collects the survival traces of a decoder's transducer searches
// (TransducerPrefixSearchOptions::trace_survival) and, when destroyed, logs
// what pruning the results needed and what cutting the lists waiting for
// later frames would have cost in results and saved in entries, so a
// policy can be chosen without rerunning. Record may be called from
// several threads.
class TransducerSurvivalSummary {
 public:
  ~TransducerSurvivalSummary();

  // One finished search and its best result, if it has one.
  void Record(const TransducerPrefixSearch &search,
              const TransducerPrefixHypothesis *best);

  // The table the destructor logs.
  std::string Report() const;

 private:
  mutable std::mutex mutex_;
  TransducerWaitHistogram waiting_;
  int64_t searches_ = 0;
  // Of each best result: on every path, and on its dominant path.
  std::vector<TransducerSurvivalNeed> needs_;
  std::vector<TransducerSurvivalNeed> dominant_needs_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_TRANSDUCER_SURVIVAL_SUMMARY_H_

// sherpa-onnx/csrc/ctc-inflight-decoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/ctc-inflight-decoder.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sherpa_onnx {

struct CtcInflightDecoder::Utterance {
  DecodeStatus status = DecodeStatus::kActive;
  int32_t nbest = 1;
  bool input_finished = false;
  std::unique_ptr<CtcPrefixSearch> search;
  // Accepted chunks not yet fully decoded; "next_frame" indexes the first.
  std::deque<std::vector<float>> chunks;
  size_t next_frame = 0;
  std::vector<CtcPrefixHypothesis> result;
};

CtcInflightDecoder::CtcInflightDecoder(
    const CtcInflightDecoderOptions &options,
    std::shared_ptr<const CtcSearchVocabulary> vocabulary,
    std::vector<std::shared_ptr<Scorer>> scorers)
    : options_(options),
      vocabulary_(std::move(vocabulary)),
      scorers_(std::move(scorers)) {
  if (options_.scorer_weights.size() != scorers_.size()) {
    throw std::invalid_argument(
        "CtcInflightDecoder: need one default weight per scorer");
  }
  for (const auto &scorer : scorers_) scorer_pointers_.push_back(scorer.get());
  PrepareScorers(*vocabulary_, scorer_pointers_,
                 options_.search.word_boundary_char);
}

CtcInflightDecoder::~CtcInflightDecoder() = default;

CtcInflightDecoder::Utterance &CtcInflightDecoder::Get(
    UtteranceHandle handle) const {
  auto found = utterances_.find(handle.id);
  if (found == utterances_.end()) {
    throw std::out_of_range("CtcInflightDecoder: no such utterance");
  }
  return *found->second;
}

std::optional<UtteranceHandle> CtcInflightDecoder::TryAdd(
    const UtteranceOptions &options) {
  if (!options.scorer_weights.empty() &&
      options.scorer_weights.size() != scorers_.size()) {
    throw std::invalid_argument(
        "CtcInflightDecoder: need one weight per scorer, or none");
  }
  if (num_active_ >= options_.max_active_utterances) return std::nullopt;

  auto utterance = std::make_unique<Utterance>();
  utterance->nbest = options.nbest;
  utterance->search = std::make_unique<CtcPrefixSearch>(
      options_.search, vocabulary_, scorer_pointers_,
      options.scorer_weights.empty() ? options_.scorer_weights
                                     : options.scorer_weights);
  UtteranceHandle handle;
  handle.id = next_handle_++;
  utterances_.emplace(handle.id, std::move(utterance));
  ++num_active_;
  return handle;
}

bool CtcInflightDecoder::TryAccept(UtteranceHandle handle,
                                   const float *log_probs, int32_t num_frames) {
  Utterance &utterance = Get(handle);
  size_t count =
      static_cast<size_t>(std::max(num_frames, 0)) * vocabulary_->tokens.size();
  size_t bytes = count * sizeof(float);
  if (bytes > options_.max_queued_input_bytes) {
    throw std::invalid_argument(
        "CtcInflightDecoder: a chunk larger than max_queued_input_bytes");
  }
  if (utterance.status != DecodeStatus::kActive || utterance.input_finished ||
      queued_bytes_ + bytes > options_.max_queued_input_bytes) {
    return false;
  }
  if (count != 0) {
    utterance.chunks.emplace_back(log_probs, log_probs + count);
    queued_bytes_ += bytes;
  }
  return true;
}

void CtcInflightDecoder::InputFinished(UtteranceHandle handle) {
  Get(handle).input_finished = true;
}

void CtcInflightDecoder::DropWork(Utterance *utterance) {
  size_t stride = vocabulary_->tokens.size() * sizeof(float);
  for (size_t i = 0; i != utterance->chunks.size(); ++i) {
    size_t frames = utterance->chunks[i].size() * sizeof(float) / stride;
    queued_bytes_ -= (frames - (i == 0 ? utterance->next_frame : 0)) * stride;
  }
  utterance->chunks.clear();
  utterance->next_frame = 0;
  utterance->search.reset();
}

void CtcInflightDecoder::Cancel(UtteranceHandle handle) {
  Utterance &utterance = Get(handle);
  if (utterance.status != DecodeStatus::kActive) return;
  DropWork(&utterance);
  utterance.status = DecodeStatus::kCancelled;
  --num_active_;
}

StepProgress CtcInflightDecoder::Step() {
  StepProgress progress;
  size_t vocab_size = vocabulary_->tokens.size();

  for (int32_t step = 0; step != options_.steps_per_utterance; ++step) {
    ready_.clear();
    for (const auto &entry : utterances_) {
      const auto &utterance = entry.second;
      if (utterance->status == DecodeStatus::kActive &&
          !utterance->chunks.empty()) {
        ready_.push_back(utterance.get());
      }
    }
    if (ready_.empty()) break;

    ready_searches_.clear();
    for (Utterance *utterance : ready_) {
      utterance->search->BeginFrame(utterance->chunks.front().data() +
                                    utterance->next_frame * vocab_size);
      ready_searches_.push_back(utterance->search.get());
    }
    RunScorerWavesTogether(ready_searches_, scorer_pointers_);
    for (Utterance *utterance : ready_) {
      utterance->search->EndFrame();
      queued_bytes_ -= vocab_size * sizeof(float);
      if (++utterance->next_frame * vocab_size ==
          utterance->chunks.front().size()) {
        utterance->chunks.pop_front();
        utterance->next_frame = 0;
      }
    }
    progress.frames_decoded += static_cast<int32_t>(ready_.size());
  }

  // Utterances whose input has ended and is used up finish together, with
  // one end-of-input call per scorer.
  std::vector<Utterance *> ending;
  std::vector<CtcPrefixSearch *> ending_searches;
  int32_t most = 0;
  for (const auto &entry : utterances_) {
    const auto &utterance = entry.second;
    if (utterance->status != DecodeStatus::kActive) {
      continue;
    }
    if (!utterance->chunks.empty()) {
      progress.has_ready_work = true;
    } else if (!utterance->input_finished) {
      progress.waiting_for_input = true;
    } else {
      ending.push_back(utterance.get());
      ending_searches.push_back(utterance->search.get());
      most = std::max(most, utterance->nbest);
    }
  }
  auto finished = CtcPrefixSearch::FinishTogether(ending_searches, most);
  for (size_t i = 0; i != ending.size(); ++i) {
    Utterance *utterance = ending[i];
    utterance->result = std::move(finished[i]);
    if (utterance->result.size() > static_cast<size_t>(utterance->nbest)) {
      utterance->result.resize(std::max(utterance->nbest, 0));
    }
    utterance->status = utterance->result.empty()
                            ? DecodeStatus::kNoAcceptedPath
                            : DecodeStatus::kComplete;
    utterance->search.reset();
    --num_active_;
  }
  return progress;
}

DecodeStatus CtcInflightDecoder::Status(UtteranceHandle handle) const {
  return Get(handle).status;
}

std::vector<CtcPrefixHypothesis> CtcInflightDecoder::GetResult(
    UtteranceHandle handle) const {
  const Utterance &utterance = Get(handle);
  if (utterance.status == DecodeStatus::kActive) {
    return {utterance.search->Best()};
  }
  return utterance.result;
}

void CtcInflightDecoder::Release(UtteranceHandle handle) {
  if (Get(handle).status == DecodeStatus::kActive) {
    throw std::invalid_argument(
        "CtcInflightDecoder: Release of an active utterance; Cancel it first");
  }
  utterances_.erase(handle.id);
}

}  // namespace sherpa_onnx

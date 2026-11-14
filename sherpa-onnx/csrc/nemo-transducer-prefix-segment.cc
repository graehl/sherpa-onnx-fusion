// sherpa-onnx/csrc/nemo-transducer-prefix-segment.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/nemo-transducer-prefix-segment.h"

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

// The prediction network's output and state after one hypothesis's tokens.
struct NeMoTransducerPrefixSegment::Prediction {
  uint32_t serial = 0;
  bool valid = false;
  Ort::Value decoder_out{nullptr};
  std::vector<Ort::Value> states;
};

// cpplint takes this constructor's wrapped lines for namespace indentation.
// NOLINTBEGIN(whitespace/indent_namespace)
NeMoTransducerPrefixSegment::NeMoTransducerPrefixSegment(
    NeMoTransducerCalls *calls, const TransducerFusion *fusion,
    const FusionConfig &config, int32_t max_active, float blank_penalty)
    : calls_(calls),
      fusion_(fusion),
      config_(config),
      max_active_(max_active),
      blank_penalty_(blank_penalty) {}
// NOLINTEND

NeMoTransducerPrefixSegment::~NeMoTransducerPrefixSegment() = default;

namespace {

Ort::Value Tokens(const std::vector<int32_t> &tokens, OrtAllocator *allocator) {
  std::array<int64_t, 2> shape = {static_cast<int64_t>(tokens.size()), 1};
  Ort::Value input =
      Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());
  std::copy(tokens.begin(), tokens.end(),
            input.GetTensorMutableData<int32_t>());
  return input;
}

}  // namespace

// The prediction for a hypothesis: computed from its parent's on first use,
// which the search asks for only after the parent's own.
auto NeMoTransducerPrefixSegment::PredictionOf(uint32_t hypothesis)
    -> Prediction & {
  if (predictions_.size() <= hypothesis) predictions_.resize(hypothesis + 1);
  Prediction &prediction = predictions_[hypothesis];
  uint32_t serial = search_->SerialOf(hypothesis);
  if (prediction.valid && prediction.serial == serial) return prediction;

  int32_t token = search_->LastTokenOf(hypothesis);
  std::vector<Ort::Value> states;
  if (token < 0) {
    token = calls_->vocab_size - 1;  // Blank begins the history.
    states = calls_->init_states();
  } else {
    const Prediction &parent = predictions_[search_->ParentOf(hypothesis)];
    for (const Ort::Value &state : parent.states) {
      states.push_back(Clone(calls_->allocator, &state));
    }
  }
  auto output = calls_->run_decoder(Tokens({token}, calls_->allocator),
                                    std::move(states));
  prediction.serial = serial;
  prediction.valid = true;
  prediction.decoder_out = std::move(output.first);
  prediction.states = std::move(output.second);
  return prediction;
}

// Computes, in one prediction-network call, the predictions the frontier
// still lacks. Each new hypothesis's parent already has its own.
void NeMoTransducerPrefixSegment::PredictTogether(
    const std::vector<uint32_t> &frontier) {
  if (!calls_->decoder_batches) return;
  OrtAllocator *allocator = calls_->allocator;
  std::vector<uint32_t> missing;
  for (uint32_t hypothesis : frontier) {
    if (predictions_.size() <= hypothesis) predictions_.resize(hypothesis + 1);
    const Prediction &prediction = predictions_[hypothesis];
    if (prediction.valid &&
        prediction.serial == search_->SerialOf(hypothesis)) {
      continue;
    }
    // The empty prefix starts from the model's initial state: left to
    // PredictionOf.
    if (search_->LastTokenOf(hypothesis) < 0) continue;
    missing.push_back(hypothesis);
  }
  auto num = static_cast<int32_t>(missing.size());
  if (num < 2) return;

  std::vector<int32_t> tokens(num);
  const auto &first_states = predictions_[search_->ParentOf(missing[0])].states;
  std::vector<Ort::Value> states;
  std::vector<std::vector<int64_t>> state_shapes;
  for (const Ort::Value &state : first_states) {
    // [layers, 1, hidden] per hypothesis; [layers, num, hidden] together.
    auto shape = state.GetTensorTypeAndShapeInfo().GetShape();
    state_shapes.push_back(shape);
    shape[1] = num;
    states.push_back(
        Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size()));
  }
  for (int32_t i = 0; i != num; ++i) {
    tokens[i] = search_->LastTokenOf(missing[i]);
    const auto &parent = predictions_[search_->ParentOf(missing[i])].states;
    for (size_t k = 0; k != states.size(); ++k) {
      int64_t layers = state_shapes[k][0], hidden = state_shapes[k][2];
      const float *from = parent[k].GetTensorData<float>();
      float *to = states[k].GetTensorMutableData<float>();
      for (int64_t layer = 0; layer != layers; ++layer) {
        std::copy(from + layer * hidden, from + (layer + 1) * hidden,
                  to + (layer * num + i) * hidden);
      }
    }
  }
  std::pair<Ort::Value, std::vector<Ort::Value>> output{
      Ort::Value{nullptr}, std::vector<Ort::Value>()};
  try {
    output = calls_->run_decoder(Tokens(tokens, allocator), std::move(states));
  } catch (const Ort::Exception &) {
    // Some exported prediction networks take one hypothesis per call.
    calls_->decoder_batches = false;
    return;
  }
  auto out_shape = output.first.GetTensorTypeAndShapeInfo().GetShape();
  size_t out_size = 1;
  for (size_t d = 1; d != out_shape.size(); ++d) out_size *= out_shape[d];
  out_shape[0] = 1;
  for (int32_t i = 0; i != num; ++i) {
    Prediction &prediction = predictions_[missing[i]];
    prediction.serial = search_->SerialOf(missing[i]);
    prediction.valid = true;
    prediction.decoder_out = Ort::Value::CreateTensor<float>(
        allocator, out_shape.data(), out_shape.size());
    const float *from = output.first.GetTensorData<float>() + i * out_size;
    std::copy(from, from + out_size,
              prediction.decoder_out.GetTensorMutableData<float>());
    prediction.states.clear();
    for (size_t k = 0; k != output.second.size(); ++k) {
      int64_t layers = state_shapes[k][0], hidden = state_shapes[k][2];
      Ort::Value one = Ort::Value::CreateTensor<float>(
          allocator, state_shapes[k].data(), state_shapes[k].size());
      const float *all = output.second[k].GetTensorData<float>();
      float *to = one.GetTensorMutableData<float>();
      for (int64_t layer = 0; layer != layers; ++layer) {
        std::copy(all + (layer * num + i) * hidden,
                  all + (layer * num + i + 1) * hidden, to + layer * hidden);
      }
      prediction.states.push_back(std::move(one));
    }
  }
}

// One joiner call for "count" hypotheses of the frontier from "first": the
// frame repeated beside each hypothesis's prediction.
void NeMoTransducerPrefixSegment::Join(const float *frame, int32_t encoder_dim,
                                       const std::vector<uint32_t> &frontier,
                                       int32_t first, int32_t count,
                                       std::vector<float> *log_probs,
                                       std::vector<float> *durations) {
  OrtAllocator *allocator = calls_->allocator;
  int32_t vocab_size = calls_->vocab_size;
  int32_t blank_id = vocab_size - 1;
  std::array<int64_t, 3> encoder_shape = {count, encoder_dim, 1};
  Ort::Value encoder_frames = Ort::Value::CreateTensor<float>(
      allocator, encoder_shape.data(), encoder_shape.size());
  float *p_encoder = encoder_frames.GetTensorMutableData<float>();
  Ort::Value decoder_outs{nullptr};
  float *p_decoder = nullptr;
  size_t decoder_size = 0;
  for (int32_t i = 0; i != count; ++i) {
    std::copy(frame, frame + encoder_dim,
              p_encoder + static_cast<size_t>(i) * encoder_dim);
    Prediction &prediction = PredictionOf(frontier[first + i]);
    if (i == 0) {
      auto shape =
          prediction.decoder_out.GetTensorTypeAndShapeInfo().GetShape();
      decoder_size = 1;
      for (int64_t d : shape) decoder_size *= static_cast<size_t>(d);
      shape[0] = count;
      decoder_outs = Ort::Value::CreateTensor<float>(allocator, shape.data(),
                                                     shape.size());
      p_decoder = decoder_outs.GetTensorMutableData<float>();
    }
    const float *one = prediction.decoder_out.GetTensorData<float>();
    std::copy(one, one + decoder_size, p_decoder + i * decoder_size);
  }
  Ort::Value logits =
      calls_->run_joiner(std::move(encoder_frames), std::move(decoder_outs));
  float *p_logits = logits.GetTensorMutableData<float>();
  size_t output_size = vocab_size + (calls_->tdt ? num_durations_ : 0);
  for (int32_t i = 0; i != count; ++i) {
    float *p_logit = p_logits + i * output_size;
    if (blank_penalty_ > 0) p_logit[blank_id] -= blank_penalty_;
    LogSoftmax(p_logit, vocab_size);
    log_probs->insert(log_probs->end(), p_logit, p_logit + vocab_size);
    if (calls_->tdt) {
      LogSoftmax(p_logit + vocab_size, num_durations_);
      durations->insert(durations->end(), p_logit + vocab_size,
                        p_logit + vocab_size + num_durations_);
    }
  }
}

void NeMoTransducerPrefixSegment::Decode(const float *encoder_out,
                                         int32_t num_frames,
                                         int32_t encoder_dim) {
  if (num_frames <= 0) return;
  int32_t vocab_size = calls_->vocab_size;
  if (search_ == nullptr) {
    TransducerPrefixSearchOptions options = MakeTransducerPrefixSearchOptions(
        config_, max_active_,
        calls_->tdt ? TransducerTopology::kTdt : TransducerTopology::kRnnt, 1);
    if (calls_->tdt) {
      // The joiner's output is the token logits followed by one logit per
      // duration, from no advance up.
      auto output = calls_->run_decoder(
          Tokens({vocab_size - 1}, calls_->allocator), calls_->init_states());
      std::array<int64_t, 3> encoder_shape = {1, encoder_dim, 1};
      Ort::Value encoder_frame = Ort::Value::CreateTensor<float>(
          calls_->allocator, encoder_shape.data(), encoder_shape.size());
      std::copy(encoder_out, encoder_out + encoder_dim,
                encoder_frame.GetTensorMutableData<float>());
      Ort::Value logit =
          calls_->run_joiner(std::move(encoder_frame), std::move(output.first));
      int32_t output_size = static_cast<int32_t>(
          logit.GetTensorTypeAndShapeInfo().GetShape().back());
      num_durations_ = output_size - vocab_size;
      if (num_durations_ < 2 ||
          num_durations_ > TransducerPrefixSearch::kMaxDuration + 1) {
        SHERPA_ONNX_LOGE("Unsupported number of TDT durations: %d",
                         num_durations_);
        SHERPA_ONNX_EXIT(-1);
      }
      options.max_duration = num_durations_ - 1;
    }
    search_ = fusion_->NewSearch(options);
  }

  int32_t first_frame = search_->Frame();
  auto run_model = [&](int32_t frame, const std::vector<uint32_t> &frontier,
                       std::vector<float> *log_probs,
                       std::vector<float> *durations) {
    const float *frame_data =
        encoder_out + static_cast<size_t>(frame - first_frame) * encoder_dim;
    auto num = static_cast<int32_t>(frontier.size());
    PredictTogether(frontier);
    if (num > 1 && calls_->joiner_batches) {
      try {
        Join(frame_data, encoder_dim, frontier, 0, num, log_probs, durations);
        return;
      } catch (const Ort::Exception &) {
        // Some exported joiners accept one hypothesis per call only.
        calls_->joiner_batches = false;
        log_probs->clear();
        durations->clear();
      }
    }
    for (int32_t i = 0; i != num; ++i) {
      Join(frame_data, encoder_dim, frontier, i, 1, log_probs, durations);
    }
  };
  DecodeTransducerFrames(search_.get(), fusion_->Scorers(),
                         first_frame + num_frames, run_model);
}

}  // namespace sherpa_onnx

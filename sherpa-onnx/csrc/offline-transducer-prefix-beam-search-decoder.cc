// sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.cc
//
// Copyright (c)  2026  graehl

#include "sherpa-onnx/csrc/offline-transducer-prefix-beam-search-decoder.h"

#include <algorithm>
#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

TransducerPrefixSearchOptions MakeTransducerPrefixSearchOptions(
    const FusionConfig &config, int32_t max_active,
    TransducerTopology topology) {
  TransducerPrefixSearchOptions options;
  options.topology = topology;
  options.max_active = max_active;
  options.max_token_candidates = config.token_candidates;
  if (config.score_beam > 0) options.beam = config.score_beam;
  options.length_bonus = config.length_bonus;
  options.length_bonus_unit = config.length_bonus_unit == "word"
                                  ? LengthBonusUnit::kWord
                                  : LengthBonusUnit::kToken;
  return options;
}

TransducerFusion::TransducerFusion(const FusionConfig &config,
                                   const SymbolTable &symbol_table,
                                   int32_t blank_id, int32_t vocab_size)
    : vocabulary_(MakeCtcSearchVocabulary(symbol_table, blank_id)),
      scorers_(CreateScorers(config.scorers)) {
  // A model may score more token IDs than the table names.
  SearchToken absent;
  absent.emit = false;
  vocabulary_->tokens.resize(vocab_size, absent);
  for (const auto &scorer : scorers_) scorer_pointers_.push_back(scorer.get());
  for (const auto &scorer : config.scorers) weights_.push_back(scorer.weight);
}

std::unique_ptr<TransducerPrefixSearch> TransducerFusion::NewSearch(
    const TransducerPrefixSearchOptions &options) const {
  return std::make_unique<TransducerPrefixSearch>(options, vocabulary_,
                                                  scorer_pointers_, weights_);
}

namespace {

// Decodes one utterance of num_frames frames. For each round,
// run_model(frame, frontier, &log_probs, &duration_log_probs) fills the
// arrays Expand takes (duration_log_probs only for kTdt).
template <class RunModel>
void DecodeTransducerUtterance(TransducerPrefixSearch *search,
                               const std::vector<Scorer *> &scorers,
                               int32_t num_frames, RunModel run_model) {
  std::vector<float> log_probs;
  std::vector<float> duration_log_probs;
  while (search->Frame() < num_frames) {
    do {
      log_probs.clear();
      duration_log_probs.clear();
      run_model(search->Frame(), search->Frontier(), &log_probs,
                &duration_log_probs);
      search->Expand(log_probs.data(), duration_log_probs.empty()
                                           ? nullptr
                                           : duration_log_probs.data());
      RunScorerWaves(search, scorers);
    } while (search->EndRound());
    search->EndFrame();
  }
}

OfflineTransducerDecoderResult ToResult(TransducerPrefixSearch *search) {
  OfflineTransducerDecoderResult result;
  auto best = search->Finish(1);
  if (best.empty()) return result;
  result.tokens.assign(best[0].tokens.begin(), best[0].tokens.end());
  result.timestamps = std::move(best[0].timestamps);
  return result;
}

int64_t LengthAt(const Ort::Value &lengths, int32_t i) {
  auto type = lengths.GetTensorTypeAndShapeInfo().GetElementType();
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return lengths.GetTensorData<int32_t>()[i];
  }
  if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    SHERPA_ONNX_LOGE("Unsupported encoder_out_length data type: %d",
                     static_cast<int32_t>(type));
    SHERPA_ONNX_EXIT(-1);
  }
  return lengths.GetTensorData<int64_t>()[i];
}

}  // namespace

OfflineTransducerPrefixBeamSearchDecoder::
    OfflineTransducerPrefixBeamSearchDecoder(OfflineTransducerModel *model,
                                             const FusionConfig &config,
                                             int32_t max_active,
                                             const SymbolTable &symbol_table,
                                             float blank_penalty)
    : model_(model),
      options_(MakeTransducerPrefixSearchOptions(
          config, max_active, TransducerTopology::kOneSymbolPerFrame)),
      fusion_(config, symbol_table, 0, model->VocabSize()),
      blank_penalty_(blank_penalty) {}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerPrefixBeamSearchDecoder::Decode(
    Ort::Value encoder_out, Ort::Value encoder_out_length,
    OfflineStream ** /*ss = nullptr*/, int32_t /*n = 0*/) {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t max_frames = static_cast<int32_t>(shape[1]);
  int32_t encoder_dim = static_cast<int32_t>(shape[2]);
  const float *p = encoder_out.GetTensorData<float>();
  int32_t vocab_size = model_->VocabSize();
  int32_t context_size = model_->ContextSize();
  OrtAllocator *allocator = model_->Allocator();

  std::vector<OfflineTransducerDecoderResult> ans(batch_size);
  for (int32_t b = 0; b != batch_size; ++b) {
    const float *frames = p + static_cast<size_t>(b) * max_frames * encoder_dim;
    auto search = fusion_.NewSearch(options_);
    auto run_model = [&](int32_t frame, const std::vector<uint32_t> &frontier,
                         std::vector<float> *log_probs,
                         std::vector<float> * /*durations*/) {
      int64_t num = static_cast<int64_t>(frontier.size());
      std::array<int64_t, 2> context_shape = {num, context_size};
      Ort::Value decoder_input = Ort::Value::CreateTensor<int64_t>(
          allocator, context_shape.data(), context_shape.size());
      int64_t *context = decoder_input.GetTensorMutableData<int64_t>();
      for (int64_t i = 0; i != num; ++i) {
        // The model's history starts with blank, padded in front with -1.
        int64_t *row = context + i * context_size;
        search->ContextOf(frontier[i], context_size, -1, row);
        int32_t length = search->LengthOf(frontier[i]);
        if (length < context_size) row[context_size - 1 - length] = 0;
      }
      Ort::Value decoder_out = model_->RunDecoder(std::move(decoder_input));

      std::array<int64_t, 2> encoder_shape = {num, encoder_dim};
      Ort::Value encoder_frames = Ort::Value::CreateTensor<float>(
          allocator, encoder_shape.data(), encoder_shape.size());
      float *dst = encoder_frames.GetTensorMutableData<float>();
      for (int64_t i = 0; i != num; ++i) {
        std::copy_n(frames + static_cast<size_t>(frame) * encoder_dim,
                    encoder_dim, dst + i * encoder_dim);
      }
      Ort::Value logit =
          model_->RunJoiner(std::move(encoder_frames), View(&decoder_out));
      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty_ > 0) {
        for (int64_t i = 0; i != num; ++i) {
          p_logit[i * vocab_size] -= blank_penalty_;
        }
      }
      LogSoftmax(p_logit, vocab_size, static_cast<int32_t>(num));
      log_probs->assign(p_logit, p_logit + num * vocab_size);
    };
    DecodeTransducerUtterance(
        search.get(), fusion_.Scorers(),
        static_cast<int32_t>(LengthAt(encoder_out_length, b)), run_model);
    ans[b] = ToResult(search.get());
  }
  return ans;
}

OfflineTransducerPrefixBeamSearchNeMoDecoder::
    OfflineTransducerPrefixBeamSearchNeMoDecoder(
        OfflineTransducerNeMoModel *model, const FusionConfig &config,
        int32_t max_active, const SymbolTable &symbol_table,
        float blank_penalty)
    : model_(model),
      config_(config),
      max_active_(max_active),
      fusion_(config, symbol_table, model->VocabSize() - 1, model->VocabSize()),
      blank_penalty_(blank_penalty) {}

namespace {

// The prediction network's output and state after one hypothesis's tokens.
struct Prediction {
  uint32_t serial = 0;
  bool valid = false;
  Ort::Value decoder_out{nullptr};
  std::vector<Ort::Value> states;
};

std::pair<Ort::Value, Ort::Value> TokenInput(int32_t token,
                                             OrtAllocator *allocator) {
  std::array<int64_t, 2> shape = {1, 1};
  Ort::Value input =
      Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());
  std::array<int64_t, 1> length_shape = {1};
  Ort::Value length = Ort::Value::CreateTensor<int32_t>(
      allocator, length_shape.data(), length_shape.size());
  input.GetTensorMutableData<int32_t>()[0] = token;
  length.GetTensorMutableData<int32_t>()[0] = 1;
  return {std::move(input), std::move(length)};
}

}  // namespace

OfflineTransducerDecoderResult
OfflineTransducerPrefixBeamSearchNeMoDecoder::DecodeOne(
    const float *encoder_out, int32_t num_frames, int32_t encoder_dim) const {
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  OrtAllocator *allocator = model_->Allocator();
  int32_t vocab_size = model_->VocabSize();
  int32_t blank_id = vocab_size - 1;
  bool tdt = model_->IsTDT();

  std::unique_ptr<TransducerPrefixSearch> search;
  std::vector<Prediction> predictions;  // By hypothesis index.

  // The prediction for a hypothesis: computed from its parent's on first
  // use, which the search asks for only after the parent's own.
  auto prediction_of = [&](uint32_t hypothesis) -> Prediction & {
    if (predictions.size() <= hypothesis) predictions.resize(hypothesis + 1);
    Prediction &prediction = predictions[hypothesis];
    uint32_t serial = search->SerialOf(hypothesis);
    if (prediction.valid && prediction.serial == serial) return prediction;

    int32_t token = search->LastTokenOf(hypothesis);
    std::vector<Ort::Value> states;
    if (token < 0) {
      token = blank_id;
      states = model_->GetDecoderInitStates(1);
    } else {
      const Prediction &parent = predictions[search->ParentOf(hypothesis)];
      for (const Ort::Value &state : parent.states) {
        states.push_back(Clone(allocator, &state));
      }
    }
    auto input = TokenInput(token, allocator);
    auto output = model_->RunDecoder(
        std::move(input.first), std::move(input.second), std::move(states));
    prediction.serial = serial;
    prediction.valid = true;
    prediction.decoder_out = std::move(output.first);
    prediction.states = std::move(output.second);
    return prediction;
  };

  std::array<int64_t, 3> encoder_shape = {1, encoder_dim, 1};
  int32_t num_durations = 0;
  auto run_model = [&](int32_t frame, const std::vector<uint32_t> &frontier,
                       std::vector<float> *log_probs,
                       std::vector<float> *durations) {
    Ort::Value encoder_frame = Ort::Value::CreateTensor(
        memory_info,
        const_cast<float *>(encoder_out) +
            static_cast<size_t>(frame) * encoder_dim,
        encoder_dim, encoder_shape.data(), encoder_shape.size());
    for (uint32_t hypothesis : frontier) {
      Prediction &prediction = prediction_of(hypothesis);
      Ort::Value logit = model_->RunJoiner(View(&encoder_frame),
                                           View(&prediction.decoder_out));
      float *p_logit = logit.GetTensorMutableData<float>();
      if (blank_penalty_ > 0) p_logit[blank_id] -= blank_penalty_;
      LogSoftmax(p_logit, vocab_size);
      log_probs->insert(log_probs->end(), p_logit, p_logit + vocab_size);
      if (tdt) {
        LogSoftmax(p_logit + vocab_size, num_durations);
        durations->insert(durations->end(), p_logit + vocab_size,
                          p_logit + vocab_size + num_durations);
      }
    }
  };

  TransducerPrefixSearchOptions options = MakeTransducerPrefixSearchOptions(
      config_, max_active_,
      tdt ? TransducerTopology::kTdt : TransducerTopology::kRnnt);
  if (tdt) {
    // The joiner's output is the token logits followed by one logit per
    // duration, from no advance up.
    auto input = TokenInput(blank_id, allocator);
    auto output =
        model_->RunDecoder(std::move(input.first), std::move(input.second),
                           model_->GetDecoderInitStates(1));
    Ort::Value encoder_frame = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(encoder_out), encoder_dim,
        encoder_shape.data(), encoder_shape.size());
    Ort::Value logit =
        model_->RunJoiner(View(&encoder_frame), View(&output.first));
    int32_t output_size = static_cast<int32_t>(
        logit.GetTensorTypeAndShapeInfo().GetShape().back());
    num_durations = output_size - vocab_size;
    if (num_durations < 2 ||
        num_durations > TransducerPrefixSearch::kMaxDuration + 1) {
      SHERPA_ONNX_LOGE("Unsupported number of TDT durations: %d",
                       num_durations);
      SHERPA_ONNX_EXIT(-1);
    }
    options.max_duration = num_durations - 1;
  }

  search = fusion_.NewSearch(options);
  DecodeTransducerUtterance(search.get(), fusion_.Scorers(), num_frames,
                            run_model);
  return ToResult(search.get());
}

std::vector<OfflineTransducerDecoderResult>
OfflineTransducerPrefixBeamSearchNeMoDecoder::Decode(
    Ort::Value encoder_out, Ort::Value encoder_out_length,
    OfflineStream ** /*ss = nullptr*/, int32_t /*n = 0*/) {
  auto shape = encoder_out.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t max_frames = static_cast<int32_t>(shape[1]);
  int32_t encoder_dim = static_cast<int32_t>(shape[2]);
  const float *p = encoder_out.GetTensorData<float>();

  std::vector<OfflineTransducerDecoderResult> ans(batch_size);
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t num_frames = static_cast<int32_t>(LengthAt(encoder_out_length, b));
    if (num_frames == 0) continue;
    ans[b] = DecodeOne(p + static_cast<size_t>(b) * max_frames * encoder_dim,
                       num_frames, encoder_dim);
  }
  return ans;
}

}  // namespace sherpa_onnx

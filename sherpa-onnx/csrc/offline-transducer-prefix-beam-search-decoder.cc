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
#include "sherpa-onnx/csrc/nemo-transducer-prefix-segment.h"
#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

TransducerPrefixSearchOptions MakeTransducerPrefixSearchOptions(
    const FusionConfig &config, int32_t max_active, TransducerTopology topology,
    int32_t context_size) {
  TransducerPrefixSearchOptions options;
  options.topology = topology;
  options.max_active = max_active;
  options.max_token_candidates = config.token_candidates;
  if (config.score_beam > 0) options.beam = config.score_beam;
  options.length_bonus = config.length_bonus;
  options.length_bonus_unit = config.length_bonus_unit == "word"
                                  ? LengthBonusUnit::kWord
                                  : LengthBonusUnit::kToken;
  if (config.length_normalize || config.modified_beam_search_rules) {
    options.rank_length_offset = context_size;
  }
  options.prune_steps = config.modified_beam_search_rules;
  options.trace_survival = config.trace_survival;
  return options;
}

int32_t UnknownTokenAsBlank(const FusionConfig &config,
                            const SymbolTable &symbol_table) {
  return config.modified_beam_search_rules && symbol_table.Contains("<unk>")
             ? symbol_table["<unk>"]
             : -1;
}

TransducerFusion::TransducerFusion(const FusionConfig &config,
                                   const SymbolTable &symbol_table,
                                   int32_t blank_id, int32_t vocab_size)
    : vocabulary_(MakeCtcSearchVocabulary(symbol_table, blank_id,
                                          ScorersWantLowercase(config))),
      scorers_(CreateScorers(config.scorers)) {
  // A model may score more token IDs than the table names.
  SearchToken absent;
  absent.emit = false;
  vocabulary_->tokens.resize(vocab_size, absent);
  for (const auto &scorer : scorers_) scorer_pointers_.push_back(scorer.get());
  for (const auto &scorer : config.scorers) weights_.push_back(scorer.weight);
  PrepareScorers(*vocabulary_, scorer_pointers_);
  if (config.trace_survival) {
    survival_ = std::make_unique<TransducerSurvivalSummary>();
  }
}

void TransducerFusion::RecordSurvival(
    const std::vector<TransducerPrefixSearch *> &searches,
    const std::vector<std::vector<TransducerPrefixHypothesis>> &results) const {
  if (survival_ == nullptr) return;
  for (size_t i = 0; i != searches.size(); ++i) {
    survival_->Record(*searches[i],
                      results[i].empty() ? nullptr : &results[i][0]);
  }
}

std::unique_ptr<TransducerPrefixSearch> TransducerFusion::NewSearch(
    const TransducerPrefixSearchOptions &options) const {
  return std::make_unique<TransducerPrefixSearch>(options, vocabulary_,
                                                  scorer_pointers_, weights_);
}

void DecodeTransducerFrames(TransducerPrefixSearch *search,
                            const std::vector<Scorer *> &scorers,
                            int32_t end_frame,
                            const TransducerRunModel &run_model) {
  std::vector<float> log_probs;
  std::vector<float> duration_log_probs;
  while (search->Frame() < end_frame) {
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

namespace {

OfflineTransducerDecoderResult ToResult(
    std::vector<TransducerPrefixHypothesis> best) {
  OfflineTransducerDecoderResult result;
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
          config, max_active, TransducerTopology::kOneSymbolPerFrame,
          model->ContextSize())),
      fusion_(config, symbol_table, 0, model->VocabSize()),
      blank_penalty_(blank_penalty) {
  options_.blank_like_token = UnknownTokenAsBlank(config, symbol_table);
}

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
  std::vector<std::unique_ptr<TransducerPrefixSearch>> searches;
  for (int32_t b = 0; b != batch_size; ++b) {
    const float *frames = p + static_cast<size_t>(b) * max_frames * encoder_dim;
    searches.push_back(fusion_.NewSearch(options_));
    TransducerPrefixSearch *search = searches.back().get();
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
    DecodeTransducerFrames(
        search, fusion_.Scorers(),
        static_cast<int32_t>(LengthAt(encoder_out_length, b)), run_model);
  }
  // The scorers' end-of-input calls serve the whole batch at once.
  std::vector<TransducerPrefixSearch *> pointers;
  for (const auto &search : searches) pointers.push_back(search.get());
  auto finished = TransducerPrefixSearch::FinishTogether(pointers, 1);
  fusion_.RecordSurvival(pointers, finished);
  for (int32_t b = 0; b != batch_size; ++b) {
    ans[b] = ToResult(std::move(finished[b]));
  }
  return ans;
}

OfflineTransducerPrefixBeamSearchNeMoDecoder::
    OfflineTransducerPrefixBeamSearchNeMoDecoder(
        OfflineTransducerNeMoModel *model, const FusionConfig &config,
        int32_t max_active, const SymbolTable &symbol_table,
        float blank_penalty)
    : config_(config),
      max_active_(max_active),
      fusion_(config, symbol_table, model->VocabSize() - 1, model->VocabSize()),
      blank_penalty_(blank_penalty),
      calls_(std::make_unique<NeMoTransducerCalls>()) {
  calls_->run_decoder = [model](Ort::Value tokens,
                                std::vector<Ort::Value> states) {
    // Every hypothesis gives the network one token.
    int64_t num = tokens.GetTensorTypeAndShapeInfo().GetShape()[0];
    std::array<int64_t, 1> shape = {num};
    Ort::Value lengths = Ort::Value::CreateTensor<int32_t>(
        model->Allocator(), shape.data(), shape.size());
    std::fill_n(lengths.GetTensorMutableData<int32_t>(), num, 1);
    return model->RunDecoder(std::move(tokens), std::move(lengths),
                             std::move(states));
  };
  calls_->run_joiner = [model](Ort::Value encoder_out, Ort::Value decoder_out) {
    return model->RunJoiner(std::move(encoder_out), std::move(decoder_out));
  };
  calls_->init_states = [model]() { return model->GetDecoderInitStates(1); };
  calls_->allocator = model->Allocator();
  calls_->vocab_size = model->VocabSize();
  calls_->tdt = model->IsTDT();
}

OfflineTransducerPrefixBeamSearchNeMoDecoder::
    ~OfflineTransducerPrefixBeamSearchNeMoDecoder() = default;

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
  std::vector<std::unique_ptr<NeMoTransducerPrefixSegment>> segments;
  std::vector<TransducerPrefixSearch *> searches;
  std::vector<int32_t> utterance;
  for (int32_t b = 0; b != batch_size; ++b) {
    int32_t num_frames = static_cast<int32_t>(LengthAt(encoder_out_length, b));
    if (num_frames == 0) continue;
    segments.push_back(std::make_unique<NeMoTransducerPrefixSegment>(
        calls_.get(), &fusion_, config_, max_active_, blank_penalty_));
    segments.back()->Decode(
        p + static_cast<size_t>(b) * max_frames * encoder_dim, num_frames,
        encoder_dim);
    searches.push_back(segments.back()->Search());
    utterance.push_back(b);
  }
  // The scorers' end-of-input calls serve the whole batch at once.
  auto finished = TransducerPrefixSearch::FinishTogether(searches, 1);
  fusion_.RecordSurvival(searches, finished);
  for (size_t i = 0; i != finished.size(); ++i) {
    ans[utterance[i]] = ToResult(std::move(finished[i]));
  }
  return ans;
}

}  // namespace sherpa_onnx

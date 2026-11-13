// sherpa-onnx/csrc/offline-recognizer-ctc-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_

#include <algorithm>
#include <array>
#include <ios>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-ctc-decoder.h"
#include "sherpa-onnx/csrc/offline-ctc-fst-decoder.h"
#include "sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-ctc-model.h"
#include "sherpa-onnx/csrc/offline-ctc-prefix-beam-search-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

inline OfflineRecognitionResult Convert(const OfflineCtcDecoderResult &src,
                                        const SymbolTable &sym_table,
                                        int32_t frame_shift_ms,
                                        int32_t subsampling_factor) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.timestamps.reserve(src.timestamps.size());

  std::string text;
  float frame_shift_s = frame_shift_ms / 1000. * subsampling_factor;

  for (int32_t i = 0; i != src.tokens.size(); ++i) {
    if (sym_table.Contains("SIL") && src.tokens[i] == sym_table["SIL"]) {
      // tdnn models from yesno have a SIL token, we should remove it.
      continue;
    }

    if (sym_table.Contains("</s>") && src.tokens[i] == sym_table["</s>"]) {
      // Skip </s> for Google MedASR
      continue;
    }
    auto sym = sym_table[src.tokens[i]];
    text.append(sym);

    if (sym.size() == 1 && (sym[0] < 0x20 || sym[0] > 0x7e)) {
      // for bpe models with byte_fallback
      // (but don't rewrite printable characters 0x20..0x7e,
      //  which collide with standard BPE units)
      std::ostringstream os;
      os << "<0x" << std::hex << std::uppercase
         << (static_cast<int32_t>(sym[0]) & 0xff) << ">";
      sym = os.str();
    }

    r.tokens.push_back(std::move(sym));
    r.timestamps.push_back(frame_shift_s * src.timestamps[i]);
  }

  if (sym_table.IsByteBpe()) {
    text = sym_table.DecodeByteBpe(text);
  }

  if (!text.empty() && text.front() == ' ') {
    text.erase(0, 1);
  }

  r.text = std::move(text);

  r.words = src.words;

  return r;
}

class OfflineRecognizerCtcImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerCtcImpl(const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(OfflineCtcModel::Create(config_.model_config)) {
    Init();
  }

  template <typename Manager>
  OfflineRecognizerCtcImpl(Manager *mgr, const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(OfflineCtcModel::Create(mgr, config_.model_config)) {
    Init();
  }

  void Init() {
    if (!config_.model_config.telespeech_ctc.empty()) {
      config_.feat_config.snip_edges = true;
      config_.feat_config.num_ceps = 40;
      config_.feat_config.feature_dim = 40;
      config_.feat_config.low_freq = 40;
      config_.feat_config.high_freq = -200;
      config_.feat_config.use_energy = false;
      config_.feat_config.normalize_samples = false;
      config_.feat_config.is_mfcc = true;
    }

    if (!config_.model_config.nemo_ctc.model.empty()) {
      if (model_->IsGigaAM()) {
        config_.feat_config.low_freq = 0;
        config_.feat_config.high_freq = 8000;
        config_.feat_config.remove_dc_offset = false;
        config_.feat_config.preemph_coeff = 0;
        config_.feat_config.window_type = "hann";
        config_.feat_config.feature_dim = 64;

        // see
        // https://github.com/salute-developers/GigaAM/blob/main/gigaam/preprocess.py#L68
        //
        // GigaAM uses n_fft 400
        config_.feat_config.round_to_power_of_two = false;
      } else {
        config_.feat_config.low_freq = 0;
        config_.feat_config.high_freq = 0;
        config_.feat_config.is_librosa = true;
        config_.feat_config.remove_dc_offset = false;
        config_.feat_config.window_type = "hann";
      }
    }

    if (!config_.model_config.dolphin.model.empty()) {
      config_.feat_config.low_freq = 0;
      config_.feat_config.high_freq = 8000;
      config_.feat_config.remove_dc_offset = false;
      config_.feat_config.dither = 0;
      config_.feat_config.preemph_coeff = 0;
      config_.feat_config.window_type = "hann";
      config_.feat_config.feature_dim = 80;
      config_.feat_config.is_librosa = true;
      config_.feat_config.frame_length_ms = 31.25;  // 16000/512 = 31.25
      config_.feat_config.snip_edges = false;
    }

    if (!config_.model_config.wenet_ctc.model.empty()) {
      // WeNet CTC models assume input samples are in the range
      // [-32768, 32767], so we set normalize_samples to false
      config_.feat_config.normalize_samples = false;
      config_.feat_config.dither = 1;
    }

    if (!config_.model_config.medasr.model.empty()) {
      config_.feat_config.low_freq = 125;
      config_.feat_config.high_freq = 7500;
      config_.feat_config.remove_dc_offset = false;
      config_.feat_config.dither = 0;
      config_.feat_config.preemph_coeff = 0;
      config_.feat_config.window_type = "hanning";
      config_.feat_config.feature_dim = 128;
      config_.feat_config.snip_edges = true;
    }

    if (!config_.model_config.fire_red_asr_ctc.model.empty()) {
      config_.feat_config.normalize_samples = false;
      config_.feat_config.high_freq = 0;
      config_.feat_config.snip_edges = true;
    }

    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    if (!config_.ctc_fst_decoder_config.graph.empty()) {
      // TODO(fangjun): Support android to read the graph from
      // asset_manager
      decoder_ = std::make_unique<OfflineCtcFstDecoder>(
          config_.ctc_fst_decoder_config);
    } else if (config_.decoding_method == "greedy_search" ||
               config_.decoding_method == "prefix_beam_search") {
      if (!symbol_table_.Contains("<blk>") &&
          !symbol_table_.Contains("<eps>") &&
          !symbol_table_.Contains("<blank>") &&
          config_.model_config.omnilingual.model.empty()) {
        // for omnilingual asr, its blank id is 0
        SHERPA_ONNX_LOGE(
            "We expect that tokens.txt contains "
            "the symbol <blk> or <eps> or <blank> and its ID.");
        SHERPA_ONNX_EXIT(-1);
      }

      int32_t blank_id = 0;
      if (symbol_table_.Contains("<blk>")) {
        blank_id = symbol_table_["<blk>"];
      } else if (symbol_table_.Contains("<eps>")) {
        // for tdnn models of the yesno recipe from icefall
        blank_id = symbol_table_["<eps>"];
      } else if (symbol_table_.Contains("<blank>")) {
        // for Wenet CTC models
        blank_id = symbol_table_["<blank>"];
      }

      if (config_.decoding_method == "prefix_beam_search") {
        try {
          decoder_ = std::make_unique<OfflineCtcPrefixBeamSearchDecoder>(
              config_.fusion, config_.max_active_paths, symbol_table_,
              blank_id);
        } catch (const std::exception &e) {
          SHERPA_ONNX_LOGE("%s", e.what());
          SHERPA_ONNX_EXIT(-1);
        }
      } else {
        decoder_ = std::make_unique<OfflineCtcGreedySearchDecoder>(blank_id);
      }
    } else {
      SHERPA_ONNX_LOGE(
          "Only greedy_search and prefix_beam_search are supported at "
          "present. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    if (config_.model_config.omnilingual.model.empty()) {
      return std::make_unique<OfflineStream>(config_.feat_config);
    } else {
      return std::make_unique<OfflineStream>(OmnilingualAsrTag{});
    }
  }

  void EncodeStreams(OfflineStream **ss, int32_t n) const override {
    std::vector<OfflineStream *> todo;
    for (int32_t i = 0; i != n; ++i) {
      if (!ss[i]->HasEncoderOutput()) {
        todo.push_back(ss[i]);
      }
    }

    if (!model_->SupportBatchProcessing() || todo.size() == 1 ||
        !config_.model_config.omnilingual.model.empty()) {
      for (OfflineStream *s : todo) {
        auto t = ForwardStream(s);
        if (!t.empty()) {
          KeepEncoderOutput(&s, 1, t);
        }
      }
    } else if (!todo.empty()) {
      auto t = ForwardStreams(todo.data(), static_cast<int32_t>(todo.size()));
      KeepEncoderOutput(todo.data(), static_cast<int32_t>(todo.size()), t);
    }
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i != n; ++i) {
      if (ss[i]->HasEncoderOutput()) {
        // Some streams were encoded ahead; encode the rest and search all.
        EncodeStreams(ss, n);
        DecodeEncodedStreams(ss, n);
        return;
      }
    }

    if (!model_->SupportBatchProcessing() || (n == 1) ||
        !config_.model_config.omnilingual.model.empty()) {
      // If the model does not support batch processing,
      // we process each stream independently.
      //
      // omnilingual asr is disabled for batch processing at present
      for (int32_t i = 0; i != n; ++i) {
        DecodeStream(ss[i]);
      }
      return;
    }

    auto t = ForwardStreams(ss, n);

    auto results = decoder_->Decode(std::move(t[0]), std::move(t[1]));

    int32_t frame_shift_ms = 10;
    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_, frame_shift_ms,
                       model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  // Runs the model on a batch of streams; returns what Forward returns.
  std::vector<Ort::Value> ForwardStreams(OfflineStream **ss, int32_t n) const {
    // Even if the omnilingual asr model can process batch input, the following
    // code does not support batching raw audio samples.

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = ss[0]->FeatureDim();

    std::vector<Ort::Value> features;
    features.reserve(n);

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int64_t> features_length_vec(n);

    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> f = ss[i]->GetFrames();

      int32_t num_frames = f.size() / feat_dim;

      model_->NormalizeFeatures(f.data(), num_frames, feat_dim);

      features_vec[i] = std::move(f);

      features_length_vec[i] = num_frames;

      std::array<int64_t, 2> shape = {num_frames, feat_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }  // for (int32_t i = 0; i != n; ++i)

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    Ort::Value x = PadSequence(model_->Allocator(), features_pointer,
                               -23.025850929940457f);
    return model_->Forward(std::move(x), std::move(x_length));
  }

  // Runs the model on one stream; returns what Forward returns, or nothing
  // when the input is too short, in which case the stream has an empty result.
  // Some models do not support batch size > 1, e.g., WeNet CTC models.
  std::vector<Ort::Value> ForwardStream(OfflineStream *s) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t feat_dim = s->FeatureDim();
    std::vector<float> f = s->GetFrames();

    // The convolutional feature extractor of omnilingual ASR models needs
    // at least 400 samples (25 ms at 16 kHz); onnxruntime throws for shorter
    // input. Empty input would also make feat_dim 0 below.
    constexpr size_t kOmnilingualMinNumSamples = 400;
    if (!config_.model_config.omnilingual.model.empty() &&
        f.size() < kOmnilingualMinNumSamples) {
      SHERPA_ONNX_LOGE(
          "Input audio is too short for omnilingual ASR model: %d samples "
          "(minimum required: %d samples, %.1f ms at 16 kHz)",
          static_cast<int>(f.size()),
          static_cast<int>(kOmnilingualMinNumSamples),
          kOmnilingualMinNumSamples / 16.0);
      OfflineRecognitionResult r;
      s->SetResult(r);
      return {};
    }

    int32_t num_frames = f.size() / feat_dim;

    model_->NormalizeFeatures(f.data(), num_frames, feat_dim);

    std::vector<int64_t> shape = {1, num_frames, feat_dim};
    if (!config_.model_config.omnilingual.model.empty()) {
      shape = {1, feat_dim};
    }

    Ort::Value x = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int64_t x_length_scalar = num_frames;
    std::array<int64_t, 1> x_length_shape = {1};
    Ort::Value x_length =
        Ort::Value::CreateTensor(memory_info, &x_length_scalar, 1,
                                 x_length_shape.data(), x_length_shape.size());

    return model_->Forward(std::move(x), std::move(x_length));
  }

  // Decode a single stream.
  void DecodeStream(OfflineStream *s) const {
    auto t = ForwardStream(s);
    if (t.empty()) {
      return;
    }

    auto results = decoder_->Decode(std::move(t[0]), std::move(t[1]));

    auto r = Convert(results[0], symbol_table_, FrameShiftMs(),
                     model_->SubsamplingFactor());
    r.text = ApplyInverseTextNormalization(std::move(r.text));
    r.text = ApplyHomophoneReplacer(std::move(r.text));
    s->SetResult(r);
  }

  int32_t FrameShiftMs() const {
    return config_.model_config.omnilingual.model.empty() ? 10 : 20;
  }

  // Moves each stream's valid frames of the model output "t" for "ss" into
  // that stream.
  static void KeepEncoderOutput(OfflineStream **ss, int32_t n,
                                const std::vector<Ort::Value> &t) {
    std::vector<int64_t> shape = t[0].GetTensorTypeAndShapeInfo().GetShape();
    auto max_frames = static_cast<int32_t>(shape[1]);
    auto dim = static_cast<int32_t>(shape[2]);
    const float *p = t[0].GetTensorData<float>();
    // The model's lengths are int64; see OfflineCtcModel::Forward.
    const int64_t *length = t[1].GetTensorData<int64_t>();
    for (int32_t i = 0; i != n; ++i) {
      auto num_frames = static_cast<int32_t>(length[i]);
      const float *begin = p + static_cast<size_t>(i) * max_frames * dim;
      ss[i]->SetEncoderOutput(
          std::vector<float>(begin,
                             begin + static_cast<size_t>(num_frames) * dim),
          num_frames);
    }
  }

  // Searches streams that EncodeStreams has processed. One it gave an empty
  // result, for input that is too short, has no output and keeps that result.
  void DecodeEncodedStreams(OfflineStream **ss, int32_t n) const {
    std::vector<OfflineStream *> encoded;
    std::vector<std::vector<float>> outputs;
    std::vector<int32_t> lengths;
    for (int32_t i = 0; i != n; ++i) {
      if (ss[i]->HasEncoderOutput()) {
        int32_t num_frames = 0;
        outputs.push_back(ss[i]->TakeEncoderOutput(&num_frames));
        lengths.push_back(num_frames);
        encoded.push_back(ss[i]);
      }
    }
    if (encoded.empty()) {
      return;
    }

    auto batch = static_cast<int32_t>(encoded.size());
    int32_t max_frames = *std::max_element(lengths.begin(), lengths.end());
    int32_t dim = 0;
    for (int32_t i = 0; i != batch && dim == 0; ++i) {
      if (lengths[i] > 0) {
        dim = static_cast<int32_t>(outputs[i].size()) / lengths[i];
      }
    }

    // onnxruntime shapes, and the lengths the decoders read, are int64.
    std::array<int64_t, 3> shape = {batch, max_frames, dim};
    Ort::Value log_probs = Ort::Value::CreateTensor<float>(
        model_->Allocator(), shape.data(), shape.size());
    float *p = log_probs.GetTensorMutableData<float>();
    size_t stride = static_cast<size_t>(max_frames) * dim;
    std::fill(p, p + batch * stride, 0.0f);
    for (int32_t i = 0; i != batch; ++i) {
      std::copy(outputs[i].begin(), outputs[i].end(), p + i * stride);
    }

    std::array<int64_t, 1> length_shape = {batch};
    Ort::Value log_probs_length = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), length_shape.data(), length_shape.size());
    std::copy(lengths.begin(), lengths.end(),
              log_probs_length.GetTensorMutableData<int64_t>());

    auto results =
        decoder_->Decode(std::move(log_probs), std::move(log_probs_length));

    for (int32_t i = 0; i != batch; ++i) {
      auto r = Convert(results[i], symbol_table_, FrameShiftMs(),
                       model_->SubsamplingFactor());
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      encoded[i]->SetResult(r);
    }
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::unique_ptr<OfflineCtcModel> model_;
  std::unique_ptr<OfflineCtcDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_

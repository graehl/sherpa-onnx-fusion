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
#include "sherpa-onnx/csrc/offline-stream-encoder.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/text-utils.h"

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

    if (!config_.model_config.w2v_bert_ctc.model.empty()) {
      // What transformers' SeamlessM4TFeatureExtractor computes: Kaldi
      // filterbanks of samples scaled to 16 bits, with no padding at the
      // edges. The model normalizes and stacks them; see
      // offline-w2v-bert-ctc-model.h
      config_.feat_config.feature_dim = 80;
      config_.feat_config.low_freq = 20;
      config_.feat_config.high_freq = 0;
      config_.feat_config.dither = 0;
      config_.feat_config.snip_edges = true;
      config_.feat_config.normalize_samples = false;
    }

    config_.feat_config.nemo_normalize_type =
        model_->FeatureNormalizationMethod();

    InitEncoder();

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
    if (encoder_) {
      encoder_->Encode(ss, n);
      return;
    }
    std::vector<OfflineStream *> todo;
    for (int32_t i = 0; i != n; ++i) {
      if (!ss[i]->HasEncoderOutput()) {
        todo.push_back(ss[i]);
      }
    }

    if (!model_->SupportBatchProcessing() || todo.size() == 1 ||
        AnyTooShortForOmnilingual(todo.data(),
                                  static_cast<int32_t>(todo.size()))) {
      for (OfflineStream *s : todo) {
        auto t = ForwardStream(s);
        if (!t.empty()) {
          KeepEncoderOutput(&s, 1, t);
        }
      }
    } else if (!todo.empty()) {
      auto t = ForwardStreams(todo.data(), static_cast<int32_t>(todo.size()));
      if (!t.empty()) {
        KeepEncoderOutput(todo.data(), static_cast<int32_t>(todo.size()), t);
      }
    }
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    if ((encoder_ && encoder_->Active()) || AnyHasEncoderOutput(ss, n)) {
      // Some streams were encoded ahead, or --encoder options apply: encode
      // the rest and search all.
      EncodeStreams(ss, n);
      DecodeEncodedStreams(ss, n);
      return;
    }

    if (!model_->SupportBatchProcessing() || (n == 1) ||
        AnyTooShortForOmnilingual(ss, n)) {
      // If the model does not support batch processing,
      // we process each stream independently.
      //
      // An omnilingual asr export batches only if it takes input_lengths;
      // see OfflineOmnilingualAsrCtcModel::SupportBatchProcessing().
      for (int32_t i = 0; i != n; ++i) {
        DecodeStream(ss[i]);
      }
      return;
    }

    auto t = ForwardStreams(ss, n);
    if (t.empty()) {
      return;  // every stream was too short and has its empty result
    }

    auto results = decoder_->Decode(std::move(t[0]), std::move(t[1]));

    int32_t frame_shift_ms = FrameShiftMs();
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
  // The shared encoder step, for every model whose input is feature frames.
  // An omnilingual model's input is its samples: it keeps its own path and
  // has no windows. A LiteRT model runs its own windows.
  void InitEncoder() {
    const auto &model_config = config_.model_config;
    if (!model_config.omnilingual.model.empty()) {
      if (model_config.encoder.window.Enabled()) {
        SHERPA_ONNX_LOGE(
            "--encoder.window.seconds does not apply to an omnilingual ASR "
            "model, whose input is samples");
        SHERPA_ONNX_EXIT(-1);
      }
      return;
    }
    OfflineEncoderConfig encoder = model_config.encoder;
    if (EndsWith(model_config.w2v_bert_ctc, ".tflite")) {
      // Its graphs have fixed lengths.
      encoder.window.seconds = 0;
      encoder.lengths.clear();
      encoder.pad_to = 0;
    }
    OfflineEncoderAdapter adapter;
    adapter.encode = [this](Ort::Value x, Ort::Value x_length) {
      return model_->Forward(std::move(x), std::move(x_length));
    };
    adapter.normalize = [this](float *f, int32_t num_frames, int32_t dim) {
      model_->NormalizeFeatures(f, num_frames, dim);
    };
    adapter.subsampling_factor = model_->SubsamplingFactor();
    adapter.batch = model_->SupportBatchProcessing();
    adapter.padding_value = -23.025850929940457f;
    adapter.rows = WindowBlend::Rows::kScores;
    adapter.allocator = model_->Allocator();
    // A zipformer export sizes its mask by the longest length; a TDNN model
    // reports every padded frame as valid.
    adapter.pads_past_longest = model_config.zipformer_ctc.model.empty() &&
                                model_config.tdnn.model.empty();
    MoveStreamNormalizationToEncoder(encoder, &config_.feat_config, &adapter);
    encoder_ = std::make_unique<OfflineStreamEncoder>(
        std::move(adapter), encoder, config_.feat_config.frame_shift_ms);
  }

  // Runs the model on a batch of streams; returns what Forward returns.
  std::vector<Ort::Value> ForwardStreams(OfflineStream **ss, int32_t n) const {
    if (!config_.model_config.omnilingual.model.empty()) {
      return ForwardOmnilingualStreams(ss, n);
    }

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

      int32_t row_dim = feat_dim;
      StackW2vBertFrames(&f, &num_frames, &row_dim);

      features_vec[i] = std::move(f);

      features_length_vec[i] = num_frames;

      std::array<int64_t, 2> shape = {num_frames, row_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }  // for (int32_t i = 0; i != n; ++i)

    // A w2v-BERT batch with no rows at all (every stream under 35 ms) would
    // reach the model empty: each stream gets an empty result and there is
    // no output. A shorter stream among longer ones is masked.
    if (!config_.model_config.w2v_bert_ctc.model.empty() &&
        *std::max_element(features_length_vec.begin(),
                          features_length_vec.end()) == 0) {
      for (int32_t i = 0; i != n; ++i) {
        ss[i]->SetResult(OfflineRecognitionResult{});
      }
      return {};
    }

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

  // The omnilingual asr model's input is the waveform: each stream's
  // "features" are its samples, as one frame. Pads the normalized samples
  // with zeros, which the model masks given the lengths.
  std::vector<Ort::Value> ForwardOmnilingualStreams(OfflineStream **ss,
                                                    int32_t n) const {
    std::vector<std::vector<float>> samples(n);
    std::vector<int64_t> lengths(n);
    int64_t max_samples = 0;
    for (int32_t i = 0; i != n; ++i) {
      samples[i] = ss[i]->GetFrames();
      lengths[i] = samples[i].size();
      model_->NormalizeFeatures(samples[i].data(), 1, lengths[i]);
      max_samples = std::max(max_samples, lengths[i]);
    }

    std::array<int64_t, 2> shape = {n, max_samples};
    Ort::Value x = Ort::Value::CreateTensor<float>(model_->Allocator(),
                                                   shape.data(), shape.size());
    float *p = x.GetTensorMutableData<float>();
    std::fill(p, p + n * max_samples, 0.0f);
    for (int32_t i = 0; i != n; ++i) {
      std::copy(samples[i].begin(), samples[i].end(), p + i * max_samples);
    }

    std::array<int64_t, 1> length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor<int64_t>(
        model_->Allocator(), length_shape.data(), length_shape.size());
    std::copy(lengths.begin(), lengths.end(),
              x_length.GetTensorMutableData<int64_t>());
    return model_->Forward(std::move(x), std::move(x_length));
  }

  // A batch cannot hold an omnilingual asr stream that is too short for the
  // model; such a call falls back to one stream at a time, where it gets an
  // empty result.
  bool AnyTooShortForOmnilingual(OfflineStream **ss, int32_t n) const {
    if (config_.model_config.omnilingual.model.empty()) {
      return false;
    }
    for (int32_t i = 0; i != n; ++i) {
      if (ss[i]->GetFrames().size() < kOmnilingualMinNumSamples) {
        return true;
      }
    }
    return false;
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
    StackW2vBertFrames(&f, &num_frames, &feat_dim);
    if (!config_.model_config.w2v_bert_ctc.model.empty() && num_frames == 0) {
      // Under 35 ms: no filterbank frame, or one, which makes no row.
      s->SetResult(OfflineRecognitionResult{});
      return {};
    }

    std::vector<int64_t> shape = {1, num_frames, feat_dim};
    if (!config_.model_config.omnilingual.model.empty()) {
      shape = {1, feat_dim};
    }

    Ort::Value x = Ort::Value::CreateTensor(memory_info, f.data(), f.size(),
                                            shape.data(), shape.size());

    int64_t x_length_scalar = num_frames;
    if (!config_.model_config.omnilingual.model.empty()) {
      // Its input is the waveform, so its length is in samples.
      x_length_scalar = feat_dim;
    }
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

  // A w2v-BERT model takes each two consecutive frames as one row of
  // 2 * feat_dim, as SeamlessM4TFeatureExtractor forms them. The frames are
  // consecutive in memory, so this only drops an odd last frame, as the
  // feature extractor does, and changes the shape.
  void StackW2vBertFrames(std::vector<float> *f, int32_t *num_frames,
                          int32_t *feat_dim) const {
    if (config_.model_config.w2v_bert_ctc.model.empty()) return;
    *num_frames /= 2;
    *feat_dim *= 2;
    f->resize(static_cast<size_t>(*num_frames) * *feat_dim);
  }

  // The convolutional feature extractor of omnilingual ASR models needs at
  // least this many samples (25 ms at 16 kHz).
  static constexpr size_t kOmnilingualMinNumSamples = 400;

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
    for (int32_t i = 0; i != n; ++i) {
      if (ss[i]->HasEncoderOutput()) {
        encoded.push_back(ss[i]);
      }
    }
    if (encoded.empty()) {
      return;
    }

    auto batch = static_cast<int32_t>(encoded.size());
    auto t = TakeEncoderOutputs(encoded.data(), batch, model_->Allocator());
    auto results = decoder_->Decode(std::move(t[0]), std::move(t[1]));

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
  // Null for an omnilingual model.
  std::unique_ptr<OfflineStreamEncoder> encoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_CTC_IMPL_H_

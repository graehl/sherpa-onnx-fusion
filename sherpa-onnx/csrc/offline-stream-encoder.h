// sherpa-onnx/csrc/offline-stream-encoder.h
//
// Copyright (c)  2026  graehl

#ifndef SHERPA_ONNX_CSRC_OFFLINE_STREAM_ENCODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_STREAM_ENCODER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/encoder-output.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/offline-encoder-window.h"
#include "sherpa-onnx/csrc/offline-stream.h"

namespace sherpa_onnx {

// What a model family supplies so that OfflineStreamEncoder can run its
// encoder ahead of the search.
struct OfflineEncoderAdapter {
  // Runs the encoder on features (N, T, C), zero or more of them padded,
  // with their lengths, int64 (N). Returns its output (N, T', D), time
  // major, and the valid frames of each, int64 (N).
  std::function<std::vector<Ort::Value>(Ort::Value features,
                                        Ort::Value lengths)>
      encode;
  // Normalizes (num_frames, dim) features in place before they are encoded;
  // empty for none.
  std::function<void(float *features, int32_t num_frames, int32_t dim)>
      normalize;
  // Input frames per output frame.
  int32_t subsampling_factor = 1;
  // false: the encoder takes one utterance per call.
  bool batch = true;
  // false: a batch must not be padded past its longest input, as for an
  // export that sizes its mask by the longest of the lengths (zipformer) or
  // reports padded frames as valid (TDNN); --encoder.lengths and
  // --encoder.pad-to are then an error.
  bool pads_past_longest = true;
  // The feature value that pads a batch.
  float padding_value = 0;
  // What the output holds, for blending windows.
  WindowBlend::Rows rows = WindowBlend::Rows::kHidden;
  OrtAllocator *allocator = nullptr;
};

// Runs an offline model's encoder for streams and keeps each stream's
// valid output frames on the stream (OfflineStream::SetEncoderOutput), for
// a later search. Streams are batched as the model allows. An utterance
// longer than --encoder.window.seconds is encoded in overlapping windows,
// which are batched like utterances, and their outputs are blended
// (topics/chunked-encoding.md).
class OfflineStreamEncoder {
 public:
  // `frame_shift_ms` is the features' frame shift. With --encoder.lengths or
  // --encoder.pad-to, the window is the shortest allowed length that holds
  // --encoder.window.seconds.
  OfflineStreamEncoder(OfflineEncoderAdapter adapter,
                       const OfflineEncoderConfig &config,
                       float frame_shift_ms);

  // Whether a long utterance is encoded in windows.
  bool Windowed() const { return schedule_.window > 0; }

  // Whether an --encoder option changes how streams are encoded (windows or
  // allowed lengths). Then a recognizer's DecodeStreams must encode through
  // Encode, not its own whole-batch path.
  bool Active() const { return Windowed() || pad_to_ > 0 || !lengths_.empty(); }

  // Encodes each of the n streams that holds no encoder output.
  void Encode(OfflineStream **ss, int32_t n) const;

 private:
  // The length an input of num_frames is padded to.
  int32_t PaddedLength(int32_t num_frames) const;

  OfflineEncoderAdapter adapter_;
  OfflineEncoderWindowConfig config_;
  // In feature frames.
  WindowSchedule schedule_;
  std::vector<int32_t> lengths_;  // ascending
  int32_t pad_to_ = 0;            // 0: no rounding
};

// With windows on, takes a normalization the stream would compute over the
// whole utterance (NeMo's per_feature, set in feat_config) out of the stream
// (FeatureExtractorConfig::nemo_normalize_in_stream) and puts it first in
// adapter->normalize, so that it runs per window or per utterance as
// --encoder.window.normalize says, and window placement always sees the same
// unnormalized features. A recognizer calls it on its own feature config
// before creating streams; every stream must then be encoded through
// OfflineStreamEncoder, which Active() guarantees.
void MoveStreamNormalizationToEncoder(const OfflineEncoderConfig &config,
                                      FeatureExtractorConfig *feat_config,
                                      OfflineEncoderAdapter *adapter);

// Whether any of the n streams holds an encoder output.
bool AnyHasEncoderOutput(OfflineStream **ss, int32_t n);

// The n streams' encoder outputs as one search input: values (n, T, D) and
// the streams' frame counts, int64 (n). When the streams hold consecutive
// items of one batch, in order, values is a view of that batch and nothing
// is copied; otherwise the items are copied into one zero-padded tensor,
// which lets a search batch be chosen apart from the encoder's. "batches"
// keeps what values may view alive; keep the whole struct until the search
// that reads it returns.
struct TakenEncoderOutputs {
  Ort::Value values{nullptr};
  Ort::Value lengths{nullptr};
  std::vector<std::shared_ptr<const EncoderOutputBatch>> batches;
};

// Moves the encoder output each of the n streams holds into a search input.
TakenEncoderOutputs TakeEncoderOutputs(OfflineStream **ss, int32_t n,
                                       OrtAllocator *allocator);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_STREAM_ENCODER_H_

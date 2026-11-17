package com.k2fsa.sherpa.onnx

// One scorer, such as a language model, fused into
// decodingMethod = "prefix_beam_search".
data class ScorerConfig(
    var type: String = "", // "kenlm" or "onnx-rnn"
    var model: String = "",
    var unit: String = "char", // "char" or "token"
    var space: String = "▁",
    var numThreads: Int = 1,
    var provider: String = "cpu",
    var weight: Float = 1.0f,
    var extra: String = "{}", // JSON object of type-specific options
)

// Scorers and search settings for decodingMethod = "prefix_beam_search".
data class FusionConfig(
    var scorers: Array<ScorerConfig> = arrayOf(),
    var lengthBonus: Float = 0.0f,
    var lengthBonusUnit: String = "token", // "token" or "word"
    var ctcAlignment: String = "sum", // "sum" or "max"
    var tokenCandidates: Int = 8,
    // A hypothesis survives a frame only within this score of the best;
    // 0 or less: no limit.
    var scoreBeam: Float = 12.0f,
)

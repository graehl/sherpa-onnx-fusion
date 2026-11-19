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
    // true: apply the scorer only to the hypotheses that survive the search,
    // as n-best rescoring does, instead of during the search.
    var rescore: Boolean = false,
    // When the scorer is destroyed, log what it measured of its own work and
    // memory (onnx-rnn: its widest model step and peak memory).
    var stats: Boolean = true,
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
    // Case of the text character scorers are given: "keep" or "lower".
    var scorerTextCase: String = "keep",
    // Transducers: rank results by score divided by output length, as
    // modified_beam_search does.
    var lengthNormalize: Boolean = false,
    // Stateless-decoder transducers: search as modified_beam_search does, so
    // that its results are reproduced, with or without scorers.
    var modifiedBeamSearchRules: Boolean = false,
    // CTC models: a text file of "token-id log-prior" lines. Each listed
    // token other than blank has ctcPriorWeight times its log prior
    // subtracted from its log probability in every frame.
    var ctcPrior: String = "",
    var ctcPriorWeight: Float = 1.0f,
    // Offline transducers: log at the end what pruning each result needed
    // and what cutting the lists waiting for later frames would cost.
    var traceSurvival: Boolean = false,
    // TDT: at every round's end also cut each list of arrivals
    // waiting past the next frame to its best waitingMaxActive within
    // waitingScoreBeam of its best.
    var pruneWaiting: Boolean = true,
    var waitingMaxActive: Int = 0, // 0: twice the max active paths
    var waitingScoreBeam: Float = 0.0f, // 0: half of scoreBeam
)

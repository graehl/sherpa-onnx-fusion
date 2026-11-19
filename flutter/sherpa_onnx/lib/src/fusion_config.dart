// Copyright (c)  2026  Xiaomi Corporation
// Shared config classes for prefix beam search with fused scorers -- no FFI,
// works on all platforms.

/// One scorer, such as a language model, fused into
/// `decodingMethod: 'prefix_beam_search'`.
///
/// For a setting these fields do not cover, such as a provider's further
/// options, use [extra] or the recognizer config's `optionsJson`.
class ScorerConfig {
  const ScorerConfig({
    this.type = '',
    this.model = '',
    this.unit = 'char',
    this.space = '▁',
    this.numThreads = 1,
    this.provider = 'cpu',
    this.weight = 1.0,
    this.extra = '{}',
    this.rescore = false,
    this.stats = true,
  });

  factory ScorerConfig.fromJson(Map<String, dynamic> json) {
    return ScorerConfig(
      type: json['type'] as String? ?? '',
      model: json['model'] as String? ?? '',
      unit: json['unit'] as String? ?? 'char',
      space: json['space'] as String? ?? '▁',
      numThreads: json['numThreads'] as int? ?? 1,
      provider: json['provider'] as String? ?? 'cpu',
      weight: (json['weight'] as num?)?.toDouble() ?? 1.0,
      extra: json['extra'] as String? ?? '{}',
      rescore: json['rescore'] as bool? ?? false,
      stats: json['stats'] as bool? ?? true,
    );
  }

  @override
  String toString() {
    return 'ScorerConfig(type: $type, model: $model, unit: $unit, space: $space, numThreads: $numThreads, provider: $provider, weight: $weight, extra: $extra, rescore: $rescore, stats: $stats)';
  }

  Map<String, dynamic> toJson() => {
        'type': type,
        'model': model,
        'unit': unit,
        'space': space,
        'numThreads': numThreads,
        'provider': provider,
        'weight': weight,
        'extra': extra,
        'rescore': rescore,
        'stats': stats,
      };

  /// Registered scorer type: `kenlm` or `onnx-rnn`.
  final String type;

  /// Path to the scorer's model.
  final String model;

  /// What the scorer reads: `char` or `token`.
  final String unit;

  /// The scorer's word-boundary unit for character scorers.
  final String space;

  /// Threads for the scorer's own model.
  final int numThreads;

  /// Execution provider for the scorer's model.
  final String provider;

  /// Search weight of the scorer's log score.
  final double weight;

  /// JSON object of type-specific options.
  final String extra;

  /// Apply the scorer only to the hypotheses that survive the search, as
  /// n-best rescoring does, instead of during the search.
  final bool rescore;

  /// When the scorer is destroyed, log what it measured of its own work and
  /// memory (`onnx-rnn`: its widest model step and peak memory).
  final bool stats;
}

/// Settings for `decodingMethod: 'prefix_beam_search'`.
class FusionConfig {
  const FusionConfig({
    this.scorers = const [],
    this.lengthBonus = 0.0,
    this.lengthBonusUnit = 'token',
    this.ctcAlignment = 'sum',
    this.tokenCandidates = 8,
    this.scoreBeam = 12.0,
    this.scorerTextCase = 'keep',
    this.lengthNormalize = false,
    this.modifiedBeamSearchRules = false,
    this.ctcPrior = '',
    this.ctcPriorWeight = 1.0,
    this.traceSurvival = false,
    this.pruneWaiting = true,
    this.waitingMaxActive = 0,
    this.waitingScoreBeam = 0.0,
  });

  factory FusionConfig.fromJson(Map<String, dynamic> json) {
    return FusionConfig(
      scorers: (json['scorers'] as List?)
              ?.map((e) => ScorerConfig.fromJson(e as Map<String, dynamic>))
              .toList() ??
          const [],
      lengthBonus: (json['lengthBonus'] as num?)?.toDouble() ?? 0.0,
      lengthBonusUnit: json['lengthBonusUnit'] as String? ?? 'token',
      ctcAlignment: json['ctcAlignment'] as String? ?? 'sum',
      tokenCandidates: json['tokenCandidates'] as int? ?? 8,
      scoreBeam: (json['scoreBeam'] as num?)?.toDouble() ?? 12.0,
      scorerTextCase: json['scorerTextCase'] as String? ?? 'keep',
      lengthNormalize: json['lengthNormalize'] as bool? ?? false,
      modifiedBeamSearchRules:
          json['modifiedBeamSearchRules'] as bool? ?? false,
      ctcPrior: json['ctcPrior'] as String? ?? '',
      ctcPriorWeight: (json['ctcPriorWeight'] as num?)?.toDouble() ?? 1.0,
      traceSurvival: json['traceSurvival'] as bool? ?? false,
      pruneWaiting: json['pruneWaiting'] as bool? ?? true,
      waitingMaxActive: json['waitingMaxActive'] as int? ?? 0,
      waitingScoreBeam:
          (json['waitingScoreBeam'] as num?)?.toDouble() ?? 0.0,
    );
  }

  @override
  String toString() {
    return 'FusionConfig(scorers: $scorers, lengthBonus: $lengthBonus, lengthBonusUnit: $lengthBonusUnit, ctcAlignment: $ctcAlignment, tokenCandidates: $tokenCandidates, scoreBeam: $scoreBeam, scorerTextCase: $scorerTextCase, lengthNormalize: $lengthNormalize, modifiedBeamSearchRules: $modifiedBeamSearchRules, ctcPrior: $ctcPrior, ctcPriorWeight: $ctcPriorWeight, traceSurvival: $traceSurvival, pruneWaiting: $pruneWaiting, waitingMaxActive: $waitingMaxActive, waitingScoreBeam: $waitingScoreBeam)';
  }

  Map<String, dynamic> toJson() => {
        'scorers': scorers.map((e) => e.toJson()).toList(),
        'lengthBonus': lengthBonus,
        'lengthBonusUnit': lengthBonusUnit,
        'ctcAlignment': ctcAlignment,
        'tokenCandidates': tokenCandidates,
        'scoreBeam': scoreBeam,
        'scorerTextCase': scorerTextCase,
        'lengthNormalize': lengthNormalize,
        'modifiedBeamSearchRules': modifiedBeamSearchRules,
        'ctcPrior': ctcPrior,
        'ctcPriorWeight': ctcPriorWeight,
        'traceSurvival': traceSurvival,
        'pruneWaiting': pruneWaiting,
        'waitingMaxActive': waitingMaxActive,
        'waitingScoreBeam': waitingScoreBeam,
      };

  /// Scorers fused with independent weights.
  final List<ScorerConfig> scorers;

  /// Added to a hypothesis's score per [lengthBonusUnit] of output.
  final double lengthBonus;

  /// What [lengthBonus] counts: `token` or `word`.
  final String lengthBonusUnit;

  /// CTC models: score an output by `sum` or `max` alignment.
  final String ctcAlignment;

  /// Most probable tokens per frame that may extend a hypothesis.
  final int tokenCandidates;

  /// A hypothesis survives a frame only within this score of the best.
  /// 0 or less: no limit.
  final double scoreBeam;

  /// Case of the text character scorers are given: `keep` or `lower`. The
  /// recognizer's output is unchanged.
  final String scorerTextCase;

  /// Transducers: rank results by score divided by output length, as
  /// `modified_beam_search` does.
  final bool lengthNormalize;

  /// Stateless-decoder transducers: search as `modified_beam_search` does,
  /// so that its results are reproduced, with or without scorers.
  final bool modifiedBeamSearchRules;

  /// CTC models: a text file of `token-id log-prior` lines. Each listed
  /// token other than blank has [ctcPriorWeight] times its log prior
  /// subtracted from its log probability in every frame.
  final String ctcPrior;

  /// Weight of [ctcPrior].
  final double ctcPriorWeight;

  /// Offline transducers: log at the end what pruning each result needed and
  /// what cutting the lists waiting for later frames would cost.
  final bool traceSurvival;

  /// Transducers: at every round's end also cut each list of arrivals
  /// waiting for a later frame to its best [waitingMaxActive] within
  /// [waitingScoreBeam] of its best.
  final bool pruneWaiting;

  /// Entries each waiting list keeps; 0: twice the max active paths.
  final int waitingMaxActive;

  /// Score gap a waiting list keeps; 0: half of [scoreBeam].
  final double waitingScoreBeam;
}

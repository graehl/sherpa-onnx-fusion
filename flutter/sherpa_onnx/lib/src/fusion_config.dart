// Copyright (c)  2026  Xiaomi Corporation
// Shared config classes for prefix beam search with fused scorers -- no FFI,
// works on all platforms.

/// One scorer, such as a language model, fused into
/// `decodingMethod: 'prefix_beam_search'`.
///
/// For a value these fields cannot express, such as a weight of 0, use the
/// recognizer config's `optionsJson`.
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
    );
  }

  @override
  String toString() {
    return 'ScorerConfig(type: $type, model: $model, unit: $unit, space: $space, numThreads: $numThreads, provider: $provider, weight: $weight, extra: $extra)';
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

  /// Search weight of the scorer's log score. 0 means 1; use `optionsJson`
  /// for a weight of 0.
  final double weight;

  /// JSON object of type-specific options.
  final String extra;
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
    );
  }

  @override
  String toString() {
    return 'FusionConfig(scorers: $scorers, lengthBonus: $lengthBonus, lengthBonusUnit: $lengthBonusUnit, ctcAlignment: $ctcAlignment, tokenCandidates: $tokenCandidates, scoreBeam: $scoreBeam, scorerTextCase: $scorerTextCase)';
  }

  Map<String, dynamic> toJson() => {
        'scorers': scorers.map((e) => e.toJson()).toList(),
        'lengthBonus': lengthBonus,
        'lengthBonusUnit': lengthBonusUnit,
        'ctcAlignment': ctcAlignment,
        'tokenCandidates': tokenCandidates,
        'scoreBeam': scoreBeam,
        'scorerTextCase': scorerTextCase,
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
  /// Negative: no limit. 0 means 12; use `optionsJson` to set 0.
  final double scoreBeam;

  /// Case of the text character scorers are given: `keep` or `lower`. The
  /// recognizer's output is unchanged.
  final String scorerTextCase;
}

// Copyright (c)  2026  Xiaomi Corporation
// Marshalling of [FusionConfig] into its C struct, shared by the offline and
// online recognizers.
import 'dart:ffi';

import 'package:ffi/ffi.dart';

import './fusion_config.dart';
import './sherpa_onnx_bindings.dart';

/// Fill [c], a struct inside a calloc'd recognizer config, from [config].
///
/// Release what this allocates with [freeFusionConfig].
void fillFusionConfig(SherpaOnnxFusionConfig c, FusionConfig config) {
  final n = config.scorers.length;
  c.numScorers = n;
  c.scorers = n == 0 ? nullptr : calloc<SherpaOnnxScorerConfig>(n);
  for (var i = 0; i < n; ++i) {
    final s = config.scorers[i];
    final p = c.scorers[i];
    p.type = s.type.toNativeUtf8();
    p.model = s.model.toNativeUtf8();
    p.unit = s.unit.toNativeUtf8();
    p.space = s.space.toNativeUtf8();
    p.numThreads = s.numThreads;
    p.provider = s.provider.toNativeUtf8();
    p.weight = s.weight;
    p.extra = s.extra.toNativeUtf8();
    p.rescore = s.rescore ? 1 : 0;
  }

  c.lengthBonus = config.lengthBonus;
  c.lengthBonusUnit = config.lengthBonusUnit.toNativeUtf8();
  c.ctcAlignment = config.ctcAlignment.toNativeUtf8();
  c.tokenCandidates = config.tokenCandidates;
  c.scoreBeam = config.scoreBeam;
  c.scorerTextCase = config.scorerTextCase.toNativeUtf8();
  c.lengthNormalize = config.lengthNormalize ? 1 : 0;
  c.modifiedBeamSearchRules = config.modifiedBeamSearchRules ? 1 : 0;
}

/// Free what [fillFusionConfig] allocated in [c].
void freeFusionConfig(SherpaOnnxFusionConfig c) {
  calloc.free(c.scorerTextCase);
  calloc.free(c.ctcAlignment);
  calloc.free(c.lengthBonusUnit);

  for (var i = 0; i < c.numScorers; ++i) {
    final p = c.scorers[i];
    calloc.free(p.extra);
    calloc.free(p.provider);
    calloc.free(p.space);
    calloc.free(p.unit);
    calloc.free(p.model);
    calloc.free(p.type);
  }
  if (c.scorers != nullptr) {
    calloc.free(c.scorers);
  }
  c.scorers = nullptr;
  c.numScorers = 0;
}

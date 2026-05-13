import '../services/threshold_config_service.dart';

class UserProfile {
  const UserProfile({
    required this.heightCm,
    required this.weightKg,
    required this.legLengthRatio,
    required this.armLengthRatio,
    required this.shoulderToHipRatio,
    required this.gender,
    required this.createdAtIso,
  });

  final double heightCm;
  final double weightKg;
  final double legLengthRatio;
  final double armLengthRatio;
  final double shoulderToHipRatio;
  final String? gender; // 'male', 'female', or null
  final String createdAtIso;

  static UserProfile defaultProfile() {
    return UserProfile(
      heightCm: 170,
      weightKg: 65,
      legLengthRatio: 0.53,
      armLengthRatio: 0.44,
      shoulderToHipRatio: 1.2,
      gender: null,
      createdAtIso: DateTime.now().toIso8601String(),
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'heightCm': heightCm,
      'weightKg': weightKg,
      'legLengthRatio': legLengthRatio,
      'armLengthRatio': armLengthRatio,
      'shoulderToHipRatio': shoulderToHipRatio,
      'gender': gender,
      'createdAtIso': createdAtIso,
    };
  }

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      heightCm: (json['heightCm'] as num?)?.toDouble() ?? 170,
      weightKg: (json['weightKg'] as num?)?.toDouble() ?? 65,
      legLengthRatio: (json['legLengthRatio'] as num?)?.toDouble() ?? 0.53,
      armLengthRatio: (json['armLengthRatio'] as num?)?.toDouble() ?? 0.44,
      shoulderToHipRatio:
          (json['shoulderToHipRatio'] as num?)?.toDouble() ?? 1.2,
      gender: json['gender'] as String?,
      createdAtIso:
          json['createdAtIso'] as String? ?? DateTime.now().toIso8601String(),
    );
  }

  /// Approximate SMPL-like shape code (beta proxy) from personal anthropometrics.
  /// This is a lightweight surrogate for on-device personalization.
  List<double> toSmplShapeCode() {
    final heightN = (heightCm - 170) / 30;
    final weightN = (weightKg - 65) / 25;
    final legN = (legLengthRatio - 0.53) / 0.08;
    final armN = (armLengthRatio - 0.44) / 0.08;
    final shoulderHipN = (shoulderToHipRatio - 1.2) / 0.35;

    return <double>[
      heightN.clamp(-2.5, 2.5).toDouble(),
      weightN.clamp(-2.5, 2.5).toDouble(),
      legN.clamp(-2.5, 2.5).toDouble(),
      armN.clamp(-2.5, 2.5).toDouble(),
      shoulderHipN.clamp(-2.5, 2.5).toDouble(),
      (heightN - weightN).clamp(-2.5, 2.5).toDouble(),
      (legN - armN).clamp(-2.5, 2.5).toDouble(),
      (0.6 * heightN + 0.4 * shoulderHipN).clamp(-2.5, 2.5).toDouble(),
      (0.5 * legN + 0.5 * shoulderHipN).clamp(-2.5, 2.5).toDouble(),
      (0.7 * weightN - 0.3 * armN).clamp(-2.5, 2.5).toDouble(),
    ];
  }
}

class PersonalizedThresholds {
  const PersonalizedThresholds({
    required this.viewTag,
    required this.squatDownAngle,
    required this.squatUpAngle,
    required this.shallowSquatMargin,
    required this.maxTorsoLeanDeg,
    required this.maxKneeInwardRatio,
    required this.maxKneeOverToe,
    required this.pushupDownAngle,
    required this.pushupUpAngle,
    required this.pushupDepthMargin,
    required this.maxHipSagDeg,
    required this.maxHipPikeDeg,
    required this.pushupHipSagOffset,
    required this.pushupHipPikeOffset,
    required this.maxElbowFlareDeg,
    required this.plankNeutralMin,
    required this.plankNeutralMax,
    required this.plankHipSagOffset,
    required this.plankHipPikeOffset,
    required this.maxNeckAngle,
  });

  final String viewTag;
  final double squatDownAngle;
  final double squatUpAngle;
  final double shallowSquatMargin;
  final double maxTorsoLeanDeg;
  final double maxKneeInwardRatio;
  final double maxKneeOverToe;
  final double pushupDownAngle;
  final double pushupUpAngle;
  final double pushupDepthMargin;
  final double maxHipSagDeg;
  final double maxHipPikeDeg;
  final double pushupHipSagOffset;
  final double pushupHipPikeOffset;
  final double maxElbowFlareDeg;
  final double plankNeutralMin;
  final double plankNeutralMax;
  final double plankHipSagOffset;
  final double plankHipPikeOffset;
  final double maxNeckAngle;

  factory PersonalizedThresholds.fromProfile(
    UserProfile profile, {
    String viewTag = 'front',
  }) {
    final resolvedViewTag = _normalizeViewTag(viewTag);
    final thresholdConfigService = ThresholdConfigService();
    final squatThresholds = thresholdConfigService.getThresholds(
      'squat',
      resolvedViewTag,
      profile.heightCm,
      profile.gender,
    );
    final pushupThresholds = thresholdConfigService.getThresholds(
      'pushup',
      resolvedViewTag,
      profile.heightCm,
      profile.gender,
    );
    final plankThresholds = thresholdConfigService.getThresholds(
      'plank',
      resolvedViewTag,
      profile.heightCm,
      profile.gender,
    );

    final bmi = profile.weightKg /
        ((profile.heightCm / 100) * (profile.heightCm / 100));
    final bmiOffset = (bmi - 22).clamp(-6.0, 8.0).toDouble();
    final legOffset =
        ((profile.legLengthRatio - 0.53) / 0.08).clamp(-2.5, 2.5).toDouble();
    final armOffset =
        ((profile.armLengthRatio - 0.44) / 0.08).clamp(-2.5, 2.5).toDouble();
    final shoulderOffset =
        ((profile.shoulderToHipRatio - 1.2) / 0.35).clamp(-2.5, 2.5).toDouble();

    final baseSquatDownAngle =
        _metricValue(squatThresholds, 'squatDownAngle', 100);
    final baseSquatUpAngle = _metricValue(squatThresholds, 'squatUpAngle', 158);
    final baseShallowSquatMargin =
        _metricValue(squatThresholds, 'shallowSquatMargin', 8);
    final baseMaxTorsoLeanDeg =
        _metricValue(squatThresholds, 'maxTorsoLeanDeg', 20);
    final baseMaxKneeInwardRatio =
        _metricValue(squatThresholds, 'maxKneeInwardRatio', 0.84);
    final baseMaxKneeOverToe =
        _metricValue(squatThresholds, 'maxKneeOverToe', 0.32);

    final basePushupDownAngle =
        _metricValue(pushupThresholds, 'pushupDownAngle', 92);
    final basePushupUpAngle =
        _metricValue(pushupThresholds, 'pushupUpAngle', 160);
    final basePushupDepthMargin =
        _metricValue(pushupThresholds, 'pushupDepthMargin', 10);
    final baseMaxHipSagDeg =
        _metricValue(pushupThresholds, 'maxHipSagDeg', 165);
    final baseMaxHipPikeDeg =
        _metricValue(pushupThresholds, 'maxHipPikeDeg', 195);
    final basePushupHipSagOffset =
        _metricValue(pushupThresholds, 'pushupHipSagOffset', -0.045);
    final basePushupHipPikeOffset =
        _metricValue(pushupThresholds, 'pushupHipPikeOffset', 0.05);
    final baseMaxElbowFlareDeg =
        _metricValue(pushupThresholds, 'maxElbowFlareDeg', 72);

    final basePlankNeutralMin =
        _metricValue(plankThresholds, 'plankNeutralMin', 168);
    final basePlankNeutralMax =
        _metricValue(plankThresholds, 'plankNeutralMax', 192);
    final basePlankHipSagOffset =
        _metricValue(plankThresholds, 'plankHipSagOffset', -0.035);
    final basePlankHipPikeOffset =
        _metricValue(plankThresholds, 'plankHipPikeOffset', 0.04);
    final baseMaxNeckAngle = _metricValue(plankThresholds, 'maxNeckAngle', 145);

    return PersonalizedThresholds(
      viewTag: resolvedViewTag,
      squatDownAngle: _clampDouble(
        baseSquatDownAngle + bmiOffset * 0.75 - legOffset * 2.4,
        85,
        115,
      ),
      squatUpAngle: _clampDouble(
        baseSquatUpAngle + legOffset * 0.5,
        150,
        170,
      ),
      shallowSquatMargin: _clampDouble(
        baseShallowSquatMargin + bmiOffset.abs() * 0.15 + legOffset.abs() * 0.5,
        5,
        14,
      ),
      maxTorsoLeanDeg: _clampDouble(
        baseMaxTorsoLeanDeg + legOffset * 1.1,
        12,
        30,
      ),
      maxKneeInwardRatio: _clampDouble(
        baseMaxKneeInwardRatio + (1.25 - profile.shoulderToHipRatio) * 0.12,
        0.76,
        0.92,
      ),
      maxKneeOverToe: _clampDouble(
        baseMaxKneeOverToe - legOffset * 0.018,
        0.20,
        0.42,
      ),
      pushupDownAngle: _clampDouble(
        basePushupDownAngle + bmiOffset * 0.35 + armOffset * 1.4,
        84,
        106,
      ),
      pushupUpAngle: _clampDouble(
        basePushupUpAngle + armOffset * 0.6,
        150,
        170,
      ),
      pushupDepthMargin: _clampDouble(
        basePushupDepthMargin + bmiOffset.abs() * 0.15 + armOffset.abs() * 0.6,
        6,
        16,
      ),
      maxHipSagDeg: _clampDouble(
        baseMaxHipSagDeg - bmiOffset * 0.3,
        158,
        176,
      ),
      maxHipPikeDeg: _clampDouble(
        baseMaxHipPikeDeg + armOffset * 1.5,
        184,
        206,
      ),
      pushupHipSagOffset: _clampDouble(
        basePushupHipSagOffset - bmiOffset * 0.0015,
        -0.09,
        -0.02,
      ),
      pushupHipPikeOffset: _clampDouble(
        basePushupHipPikeOffset + armOffset * 0.004,
        0.02,
        0.09,
      ),
      maxElbowFlareDeg: _clampDouble(
        baseMaxElbowFlareDeg + shoulderOffset * 4.5,
        55,
        85,
      ),
      plankNeutralMin: _clampDouble(
        basePlankNeutralMin - bmiOffset * 0.25,
        160,
        178,
      ),
      plankNeutralMax: _clampDouble(
        basePlankNeutralMax + armOffset * 0.8,
        182,
        200,
      ),
      plankHipSagOffset: _clampDouble(
        basePlankHipSagOffset - bmiOffset * 0.0012,
        -0.08,
        -0.015,
      ),
      plankHipPikeOffset: _clampDouble(
        basePlankHipPikeOffset + armOffset * 0.003,
        0.015,
        0.08,
      ),
      maxNeckAngle: _clampDouble(
        baseMaxNeckAngle - (profile.heightCm - 170) * 0.04,
        130,
        160,
      ),
    );
  }

  static String _normalizeViewTag(String viewTag) {
    switch (viewTag.trim().toLowerCase()) {
      case 'side':
        return 'side';
      case 'oblique':
        return 'oblique';
      default:
        return 'front';
    }
  }
}

double _metricValue(Map<String, double> values, String key, double fallback) {
  return values[key] ?? fallback;
}

double _clampDouble(num value, double min, double max) {
  return value.clamp(min, max).toDouble();
}

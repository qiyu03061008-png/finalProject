class ThresholdMetricProfile {
  const ThresholdMetricProfile({
    required this.selected,
    this.p10,
    this.p50,
    this.p90,
    this.sampleCount = 0,
    this.selectionRule,
    this.sourceMetric,
  });

  final double selected;
  final double? p10;
  final double? p50;
  final double? p90;
  final int sampleCount;
  final String? selectionRule;
  final String? sourceMetric;

  factory ThresholdMetricProfile.fromJson(dynamic json) {
    if (json is num) {
      return ThresholdMetricProfile(selected: json.toDouble());
    }
    if (json is! Map<String, dynamic>) {
      throw ArgumentError('Invalid threshold metric payload: $json');
    }
    final selectedValue = json['selected'] ?? json['value'] ?? json['default'];
    if (selectedValue is! num) {
      throw ArgumentError('Threshold metric missing selected value: $json');
    }
    return ThresholdMetricProfile(
      selected: selectedValue.toDouble(),
      p10: (json['p10'] as num?)?.toDouble(),
      p50: (json['p50'] as num?)?.toDouble(),
      p90: (json['p90'] as num?)?.toDouble(),
      sampleCount: (json['sampleCount'] as num?)?.toInt() ??
          (json['sample_count'] as num?)?.toInt() ??
          0,
      selectionRule:
          json['selectionRule'] as String? ?? json['selection_rule'] as String?,
      sourceMetric:
          json['sourceMetric'] as String? ?? json['source_metric'] as String?,
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'selected': selected,
      if (p10 != null) 'p10': p10,
      if (p50 != null) 'p50': p50,
      if (p90 != null) 'p90': p90,
      'sampleCount': sampleCount,
      if (selectionRule != null) 'selectionRule': selectionRule,
      if (sourceMetric != null) 'sourceMetric': sourceMetric,
    };
  }
}

class ThresholdBucketProfile {
  const ThresholdBucketProfile({
    required this.metrics,
    this.metadata = const <String, dynamic>{},
  });

  final Map<String, ThresholdMetricProfile> metrics;
  final Map<String, dynamic> metadata;

  factory ThresholdBucketProfile.fromJson(dynamic json) {
    if (json is! Map<String, dynamic>) {
      throw ArgumentError('Invalid threshold bucket payload: $json');
    }

    final metricsJson = json['metrics'];
    if (metricsJson is Map<String, dynamic>) {
      return ThresholdBucketProfile(
        metrics: metricsJson.map(
          (key, value) => MapEntry(
            key,
            ThresholdMetricProfile.fromJson(value),
          ),
        ),
        metadata: Map<String, dynamic>.from(
          (json['metadata'] as Map?)?.cast<String, dynamic>() ??
              const <String, dynamic>{},
        ),
      );
    }

    // Legacy flat leaf map support.
    final flatMetrics = <String, ThresholdMetricProfile>{};
    json.forEach((key, value) {
      if (value is num) {
        flatMetrics[key] = ThresholdMetricProfile(selected: value.toDouble());
      }
    });
    return ThresholdBucketProfile(metrics: flatMetrics);
  }

  Map<String, double> selectedValues() {
    return metrics.map(
      (key, value) => MapEntry(key, value.selected),
    );
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'metrics': metrics.map((key, value) => MapEntry(key, value.toJson())),
      if (metadata.isNotEmpty) 'metadata': metadata,
    };
  }
}

class ThresholdProfile {
  const ThresholdProfile({
    required this.schemaVersion,
    required this.generatedAtIso,
    required this.exercises,
  });

  final int schemaVersion;
  final String generatedAtIso;
  final Map<String, Map<String, Map<String, ThresholdBucketProfile>>> exercises;

  factory ThresholdProfile.fromJson(Map<String, dynamic> json) {
    if (json.containsKey('exercises')) {
      final exercises =
          <String, Map<String, Map<String, ThresholdBucketProfile>>>{};
      final exercisesJson = json['exercises'] as Map<String, dynamic>;
      exercisesJson.forEach((exercise, exerciseValue) {
        final viewMap = <String, Map<String, ThresholdBucketProfile>>{};
        (exerciseValue as Map<String, dynamic>).forEach((view, viewValue) {
          final groupMap = <String, ThresholdBucketProfile>{};
          (viewValue as Map<String, dynamic>).forEach((group, groupValue) {
            groupMap[group] = ThresholdBucketProfile.fromJson(groupValue);
          });
          viewMap[view] = groupMap;
        });
        exercises[exercise] = viewMap;
      });
      return ThresholdProfile(
        schemaVersion: (json['schemaVersion'] as num?)?.toInt() ?? 2,
        generatedAtIso: json['generatedAtIso'] as String? ??
            json['generated_at'] as String? ??
            DateTime.now().toIso8601String(),
        exercises: exercises,
      );
    }

    // Legacy schema support.
    final exercises =
        <String, Map<String, Map<String, ThresholdBucketProfile>>>{};
    json.forEach((exercise, exerciseValue) {
      final viewMap = <String, Map<String, ThresholdBucketProfile>>{};
      (exerciseValue as Map<String, dynamic>).forEach((view, viewValue) {
        final groupMap = <String, ThresholdBucketProfile>{};
        (viewValue as Map<String, dynamic>).forEach((group, groupValue) {
          groupMap[group] = ThresholdBucketProfile.fromJson(groupValue);
        });
        viewMap[view] = groupMap;
      });
      exercises[exercise] = viewMap;
    });

    return ThresholdProfile(
      schemaVersion: 1,
      generatedAtIso: DateTime.now().toIso8601String(),
      exercises: exercises,
    );
  }

  factory ThresholdProfile.defaultProfile() {
    return ThresholdProfile(
      schemaVersion: 2,
      generatedAtIso: DateTime.now().toIso8601String(),
      exercises: _wrapDefaultMetrics(_defaultSelectedThresholds),
    );
  }

  ThresholdBucketProfile? resolveBucket({
    required String exercise,
    required String view,
    required List<String> groups,
  }) {
    final viewMap = exercises[exercise];
    if (viewMap == null) {
      return null;
    }

    final candidateViews = <String>[
      view,
      if (view != 'default') 'default',
      ...viewMap.keys.where((element) => element != view && element != 'default'),
    ];

    for (final candidateView in candidateViews) {
      final groupMap = viewMap[candidateView];
      if (groupMap == null) {
        continue;
      }
      for (final group in groups) {
        final bucket = groupMap[group];
        if (bucket != null) {
          return bucket;
        }
      }
      final defaultBucket = groupMap['default'];
      if (defaultBucket != null) {
        return defaultBucket;
      }
    }
    return null;
  }

  Map<String, double> resolveThresholds({
    required String exercise,
    required String view,
    required List<String> groups,
  }) {
    return resolveBucket(exercise: exercise, view: view, groups: groups)
            ?.selectedValues() ??
        const <String, double>{};
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'schemaVersion': schemaVersion,
      'generatedAtIso': generatedAtIso,
      'exercises': exercises.map(
        (exercise, viewMap) => MapEntry(
          exercise,
          viewMap.map(
            (view, groupMap) => MapEntry(
              view,
              groupMap.map(
                (group, bucket) => MapEntry(group, bucket.toJson()),
              ),
            ),
          ),
        ),
      ),
    };
  }
}

Map<String, Map<String, Map<String, ThresholdBucketProfile>>> _wrapDefaultMetrics(
  Map<String, Map<String, Map<String, Map<String, double>>>> input,
) {
  return input.map(
    (exercise, viewMap) => MapEntry(
      exercise,
      viewMap.map(
        (view, groupMap) => MapEntry(
          view,
          groupMap.map(
            (group, metrics) => MapEntry(
              group,
              ThresholdBucketProfile(
                metrics: metrics.map(
                  (metric, value) => MapEntry(
                    metric,
                    ThresholdMetricProfile(
                      selected: value,
                      sampleCount: 0,
                      selectionRule: 'fallback_default',
                    ),
                  ),
                ),
                metadata: const <String, dynamic>{
                  'fallbackOnly': true,
                },
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

const Map<String, Map<String, Map<String, Map<String, double>>>>
    _defaultSelectedThresholds = <String,
        Map<String, Map<String, Map<String, double>>>>{
  'squat': <String, Map<String, Map<String, double>>>{
    'front': <String, Map<String, double>>{
      'default': <String, double>{
        'squatDownAngle': 100,
        'squatUpAngle': 158,
        'shallowSquatMargin': 8,
        'maxTorsoLeanDeg': 20,
        'maxKneeInwardRatio': 0.84,
        'maxKneeOverToe': 0.32,
      },
    },
    'side': <String, Map<String, double>>{
      'default': <String, double>{
        'squatDownAngle': 100,
        'squatUpAngle': 158,
        'shallowSquatMargin': 8,
        'maxTorsoLeanDeg': 20,
        'maxKneeInwardRatio': 0.84,
        'maxKneeOverToe': 0.32,
      },
    },
    'oblique': <String, Map<String, double>>{
      'default': <String, double>{
        'squatDownAngle': 100,
        'squatUpAngle': 158,
        'shallowSquatMargin': 8,
        'maxTorsoLeanDeg': 20,
        'maxKneeInwardRatio': 0.84,
        'maxKneeOverToe': 0.32,
      },
    },
  },
  'pushup': <String, Map<String, Map<String, double>>>{
    'front': <String, Map<String, double>>{
      'default': <String, double>{
        'pushupDownAngle': 92,
        'pushupUpAngle': 160,
        'pushupDepthMargin': 10,
        'maxHipSagDeg': 165,
        'maxHipPikeDeg': 195,
        'pushupHipSagOffset': -0.045,
        'pushupHipPikeOffset': 0.05,
        'maxElbowFlareDeg': 72,
      },
    },
    'side': <String, Map<String, double>>{
      'default': <String, double>{
        'pushupDownAngle': 92,
        'pushupUpAngle': 160,
        'pushupDepthMargin': 10,
        'maxHipSagDeg': 165,
        'maxHipPikeDeg': 195,
        'pushupHipSagOffset': -0.045,
        'pushupHipPikeOffset': 0.05,
        'maxElbowFlareDeg': 72,
      },
    },
    'oblique': <String, Map<String, double>>{
      'default': <String, double>{
        'pushupDownAngle': 92,
        'pushupUpAngle': 160,
        'pushupDepthMargin': 10,
        'maxHipSagDeg': 165,
        'maxHipPikeDeg': 195,
        'pushupHipSagOffset': -0.045,
        'pushupHipPikeOffset': 0.05,
        'maxElbowFlareDeg': 72,
      },
    },
  },
  'plank': <String, Map<String, Map<String, double>>>{
    'front': <String, Map<String, double>>{
      'default': <String, double>{
        'plankNeutralMin': 168,
        'plankNeutralMax': 192,
        'plankHipSagOffset': -0.035,
        'plankHipPikeOffset': 0.04,
        'maxNeckAngle': 145,
      },
    },
    'side': <String, Map<String, double>>{
      'default': <String, double>{
        'plankNeutralMin': 168,
        'plankNeutralMax': 192,
        'plankHipSagOffset': -0.035,
        'plankHipPikeOffset': 0.04,
        'maxNeckAngle': 145,
      },
    },
    'oblique': <String, Map<String, double>>{
      'default': <String, double>{
        'plankNeutralMin': 168,
        'plankNeutralMax': 192,
        'plankHipSagOffset': -0.035,
        'plankHipPikeOffset': 0.04,
        'maxNeckAngle': 145,
      },
    },
  },
};

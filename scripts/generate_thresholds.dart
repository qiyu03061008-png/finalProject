import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:fitness_pose_app/models/pose_landmark.dart';
import 'package:fitness_pose_app/models/threshold_profile.dart';
import 'package:fitness_pose_app/services/pose_metric_calculator.dart';

Future<void> main() async {
  const generator = ThresholdProfileGenerator();
  final profile = await generator.generate();
  final outputFile = File('assets/threshold_profile.json');

  await outputFile.parent.create(recursive: true);
  const encoder = JsonEncoder.withIndent('  ');
  await outputFile.writeAsString(
    '${encoder.convert(profile.toJson())}\n',
    flush: true,
  );

  stdout.writeln('Threshold profile written to ${outputFile.path}');
}

class ThresholdProfileGenerator {
  const ThresholdProfileGenerator({PoseMetricCalculator? calculator})
      : _calculator = calculator ?? const PoseMetricCalculator();

  final PoseMetricCalculator _calculator;

  static const String _datasetDirectoryPath = 'data/dataset';
  static const String _mergedDatasetPath = 'data/dataset/fitness_dataset.jsonl';

  static const List<String> _fitCoachInputCandidates = <String>[
    'data/qevd_fitcoach/processed/fitcoach_landmarks_app.jsonl',
    'data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl',
  ];

  Future<ThresholdProfile> generate() async {
    final baseProfile = ThresholdProfile.defaultProfile();
    final exercises = _cloneExercises(baseProfile.exercises);
    final buckets = <_BucketKey, _BucketAccumulator>{};

    for (final inputPath in _inputPathsForRun()) {
      final stats = await _ingestFile(inputPath, buckets);
      stdout.writeln(stats.summaryLine());
    }

    for (final entry in buckets.entries) {
      final exerciseViews = exercises.putIfAbsent(
        entry.key.exercise,
        () => <String, Map<String, ThresholdBucketProfile>>{},
      );
      final viewGroups = exerciseViews.putIfAbsent(
        entry.key.view,
        () => <String, ThresholdBucketProfile>{},
      );
      final fallbackBucket = viewGroups[entry.key.group] ??
          viewGroups['default'] ??
          _defaultBucketFor(entry.key.exercise, entry.key.view, baseProfile);
      viewGroups[entry.key.group] = entry.value.buildProfile(fallbackBucket);
    }

    return ThresholdProfile(
      schemaVersion: 2,
      generatedAtIso: DateTime.now().toIso8601String(),
      exercises: exercises,
    );
  }

  Future<_IngestionStats> _ingestFile(
    String inputPath,
    Map<_BucketKey, _BucketAccumulator> buckets,
  ) async {
    final file = File(inputPath);
    final stats = _IngestionStats(path: inputPath);

    if (!file.existsSync()) {
      stats.missing = true;
      return stats;
    }

    final lines = await file.readAsLines();
    for (final rawLine in lines) {
      final line = rawLine.trim();
      if (line.isEmpty) {
        continue;
      }
      stats.totalLines += 1;

      final record = _decodeJsonObject(line);
      if (record == null) {
        stats.invalidJson += 1;
        continue;
      }

      final exercise = _normalizeExercise(record['exercise_type'] as String?);
      if (exercise == null) {
        stats.unsupportedExercise += 1;
        continue;
      }

      final pose = _buildPose(record);
      if (pose == null) {
        stats.skippedNoLandmarks += 1;
        continue;
      }

      final snapshot = _calculator.calculate(pose);
      final viewTag = _normalizeView(record['view_tag'] as String?) ??
          snapshot.inferredViewTag;
      final gender = _normalizeGender(_readGender(record));
      final heightCm = _readHeightCm(record);
      final heightGroup = _heightGroup(heightCm);
      final sourceName = _sourceName(record, inputPath);
      final poseErrors = _poseErrors(record);
      final qualityTag =
          (record['quality_tag'] as String?)?.trim().toLowerCase();
      final isErrorSample = qualityTag == 'error' || poseErrors.isNotEmpty;
      final groups = _groupsFor(gender: gender, heightGroup: heightGroup);

      for (final group in groups) {
        final bucket = buckets.putIfAbsent(
          _BucketKey(exercise: exercise, view: viewTag, group: group),
          () => _BucketAccumulator(),
        );
        bucket.addObservation(
          metrics: snapshot.metrics,
          sourceName: sourceName,
          isErrorSample: isErrorSample,
          poseErrors: poseErrors,
        );
      }

      stats.usableSamples += 1;
      if (isErrorSample) {
        stats.errorSamples += 1;
      } else {
        stats.standardSamples += 1;
      }
    }

    return stats;
  }

  Map<String, Map<String, Map<String, ThresholdBucketProfile>>> _cloneExercises(
    Map<String, Map<String, Map<String, ThresholdBucketProfile>>> source,
  ) {
    return source.map(
      (exercise, viewMap) => MapEntry(
        exercise,
        viewMap.map(
          (view, groupMap) => MapEntry(
            view,
            Map<String, ThresholdBucketProfile>.from(groupMap),
          ),
        ),
      ),
    );
  }

  ThresholdBucketProfile _defaultBucketFor(
    String exercise,
    String view,
    ThresholdProfile baseProfile,
  ) {
    return baseProfile.resolveBucket(
          exercise: exercise,
          view: view,
          groups: const <String>['default'],
        ) ??
        const ThresholdBucketProfile(
            metrics: <String, ThresholdMetricProfile>{});
  }

  List<String> _inputPathsForRun() {
    final resolved = <String>[];
    final mergedDatasetFile = File(_mergedDatasetPath);
    if (mergedDatasetFile.existsSync()) {
      resolved.add(_mergedDatasetPath);
    } else {
      final datasetDir = Directory(_datasetDirectoryPath);
      if (datasetDir.existsSync()) {
        final datasetFiles = datasetDir
            .listSync()
            .whereType<File>()
            .where((file) => file.path.toLowerCase().endsWith('.jsonl'))
            .where((file) {
          final name = file.uri.pathSegments.isEmpty
              ? file.path.toLowerCase()
              : file.uri.pathSegments.last.toLowerCase();
          return name.startsWith('fitness_dataset');
        }).toList()
          ..sort((a, b) => a.path.compareTo(b.path));
        resolved.addAll(datasetFiles.map((file) => file.path));
      }
    }
    for (final candidate in _fitCoachInputCandidates) {
      if (File(candidate).existsSync()) {
        resolved.add(candidate);
        break;
      }
    }
    return resolved;
  }
}

class _BucketAccumulator {
  int standardSampleCount = 0;
  int errorSampleCount = 0;
  final Map<String, int> errorCounts = <String, int>{};
  final Map<String, List<double>> metricValues = <String, List<double>>{};
  final Set<String> sources = <String>{};

  void addObservation({
    required Map<String, double> metrics,
    required String sourceName,
    required bool isErrorSample,
    required List<String> poseErrors,
  }) {
    sources.add(sourceName);
    if (isErrorSample) {
      errorSampleCount += 1;
      for (final poseError in poseErrors) {
        errorCounts[poseError] = (errorCounts[poseError] ?? 0) + 1;
      }
      return;
    }

    standardSampleCount += 1;
    metrics.forEach((key, value) {
      if (value.isFinite && !value.isNaN) {
        metricValues.putIfAbsent(key, () => <double>[]).add(value);
      }
    });
  }

  ThresholdBucketProfile buildProfile(ThresholdBucketProfile fallbackBucket) {
    final metrics = <String, ThresholdMetricProfile>{};

    for (final entry in fallbackBucket.metrics.entries) {
      metrics[entry.key] = _buildMetricProfile(entry.key, entry.value);
    }

    return ThresholdBucketProfile(
      metrics: metrics,
      metadata: <String, dynamic>{
        'standardSampleCount': standardSampleCount,
        'errorSampleCount': errorSampleCount,
        'errorCounts': errorCounts,
        'sources': sources.toList()..sort(),
      },
    );
  }

  ThresholdMetricProfile _buildMetricProfile(
    String thresholdKey,
    ThresholdMetricProfile fallbackMetric,
  ) {
    switch (thresholdKey) {
      case 'squatDownAngle':
        return _buildLowerTransition(
          sourceMetric: 'squatKneeAngle',
          fallbackMetric: fallbackMetric,
          minValue: 85,
          maxValue: 115,
          mix: 0.35,
        );
      case 'squatUpAngle':
        return _buildUpperTransition(
          sourceMetric: 'squatKneeAngle',
          fallbackMetric: fallbackMetric,
          minValue: 150,
          maxValue: 170,
          mix: 0.65,
        );
      case 'shallowSquatMargin':
        return _buildMarginFromSpread(
          sourceMetric: 'squatKneeAngle',
          fallbackMetric: fallbackMetric,
          minValue: 5,
          maxValue: 14,
          scale: 0.20,
        );
      case 'maxTorsoLeanDeg':
        return _buildUpperBound(
          sourceMetric: 'squatTorsoLeanDeg',
          fallbackMetric: fallbackMetric,
          minValue: 12,
          maxValue: 22,
        );
      case 'maxKneeValgusAngleDeg':
        return _buildLowerBound(
          sourceMetric: 'squatKneeValgusAngle',
          fallbackMetric: fallbackMetric,
          minValue: 162,
          maxValue: 172,
        );
      case 'maxShankLeanDeg':
        return _buildUpperBound(
          sourceMetric: 'squatShankLeanDeg',
          fallbackMetric: fallbackMetric,
          minValue: 22,
          maxValue: 32,
        );
      case 'pushupDownAngle':
        return _buildLowerTransition(
          sourceMetric: 'pushupElbowAngle',
          fallbackMetric: fallbackMetric,
          minValue: 84,
          maxValue: 106,
          mix: 0.35,
        );
      case 'pushupUpAngle':
        return _buildUpperTransition(
          sourceMetric: 'pushupElbowAngle',
          fallbackMetric: fallbackMetric,
          minValue: 150,
          maxValue: 170,
          mix: 0.65,
        );
      case 'pushupDepthMargin':
        return _buildMarginFromSpread(
          sourceMetric: 'pushupElbowAngle',
          fallbackMetric: fallbackMetric,
          minValue: 6,
          maxValue: 12,
          scale: 0.25,
        );
      case 'maxHipSagDeg':
        return _buildAngleAroundStraight(
          sourceMetric: 'pushupBodyLineDeviation',
          fallbackMetric: fallbackMetric,
          upperSide: false,
          minValue: 155,
          maxValue: 180,
        );
      case 'maxHipPikeDeg':
        return _buildAngleAroundStraight(
          sourceMetric: 'pushupBodyLineDeviation',
          fallbackMetric: fallbackMetric,
          upperSide: true,
          minValue: 180,
          maxValue: 205,
        );
      case 'pushupHipSagOffset':
        return _buildLowerBound(
          sourceMetric: 'pushupHipOffset',
          fallbackMetric: fallbackMetric,
          minValue: -0.10,
          maxValue: -0.01,
        );
      case 'pushupHipPikeOffset':
        return _buildUpperBound(
          sourceMetric: 'pushupHipOffset',
          fallbackMetric: fallbackMetric,
          minValue: 0.01,
          maxValue: 0.10,
        );
      case 'maxElbowFlareDeg':
        return _buildUpperBound(
          sourceMetric: 'pushupElbowFlareDeg',
          fallbackMetric: fallbackMetric,
          minValue: 45,
          maxValue: 78,
        );
      case 'plankNeutralMin':
        return _buildAngleAroundStraight(
          sourceMetric: 'plankBodyLineDeviation',
          fallbackMetric: fallbackMetric,
          upperSide: false,
          minValue: 155,
          maxValue: 180,
        );
      case 'plankNeutralMax':
        return _buildAngleAroundStraight(
          sourceMetric: 'plankBodyLineDeviation',
          fallbackMetric: fallbackMetric,
          upperSide: true,
          minValue: 180,
          maxValue: 205,
        );
      case 'plankHipSagOffset':
        return _buildLowerBound(
          sourceMetric: 'plankHipOffset',
          fallbackMetric: fallbackMetric,
          minValue: -0.10,
          maxValue: -0.01,
        );
      case 'plankHipPikeOffset':
        return _buildUpperBound(
          sourceMetric: 'plankHipOffset',
          fallbackMetric: fallbackMetric,
          minValue: 0.01,
          maxValue: 0.10,
        );
      case 'maxNeckAngle':
        return _buildLowerBound(
          sourceMetric: 'plankNeckAngle',
          fallbackMetric: fallbackMetric,
          minValue: 140,
          maxValue: 155,
        );
      default:
        return ThresholdMetricProfile(
          selected: fallbackMetric.selected,
          p10: fallbackMetric.p10,
          p50: fallbackMetric.p50,
          p90: fallbackMetric.p90,
          sampleCount: fallbackMetric.sampleCount,
          selectionRule: fallbackMetric.selectionRule ?? 'fallback_default',
          sourceMetric: fallbackMetric.sourceMetric,
        );
    }
  }

  ThresholdMetricProfile _buildLowerTransition({
    required String sourceMetric,
    required ThresholdMetricProfile fallbackMetric,
    required double minValue,
    required double maxValue,
    required double mix,
  }) {
    final stats = _statsFor(sourceMetric);
    if (stats == null) {
      return _fallbackMetric(fallbackMetric, sourceMetric);
    }
    final dataSelected = _lerp(stats.p10, stats.p50, mix);
    return _selectedFromData(
      stats: stats,
      fallbackMetric: fallbackMetric,
      sourceMetric: sourceMetric,
      selectionRule: 'blend_lower_transition',
      dataSelected: dataSelected,
      minValue: minValue,
      maxValue: maxValue,
    );
  }

  ThresholdMetricProfile _buildUpperTransition({
    required String sourceMetric,
    required ThresholdMetricProfile fallbackMetric,
    required double minValue,
    required double maxValue,
    required double mix,
  }) {
    final stats = _statsFor(sourceMetric);
    if (stats == null) {
      return _fallbackMetric(fallbackMetric, sourceMetric);
    }
    final dataSelected = _lerp(stats.p50, stats.p90, mix);
    return _selectedFromData(
      stats: stats,
      fallbackMetric: fallbackMetric,
      sourceMetric: sourceMetric,
      selectionRule: 'blend_upper_transition',
      dataSelected: dataSelected,
      minValue: minValue,
      maxValue: maxValue,
    );
  }

  ThresholdMetricProfile _buildMarginFromSpread({
    required String sourceMetric,
    required ThresholdMetricProfile fallbackMetric,
    required double minValue,
    required double maxValue,
    required double scale,
  }) {
    final stats = _statsFor(sourceMetric);
    if (stats == null) {
      return _fallbackMetric(fallbackMetric, sourceMetric);
    }
    final dataSelected = (stats.p50 - stats.p10).abs() * scale;
    return _selectedFromData(
      stats: stats,
      fallbackMetric: fallbackMetric,
      sourceMetric: sourceMetric,
      selectionRule: 'blend_spread_margin',
      dataSelected: dataSelected,
      minValue: minValue,
      maxValue: maxValue,
    );
  }

  ThresholdMetricProfile _buildUpperBound({
    required String sourceMetric,
    required ThresholdMetricProfile fallbackMetric,
    required double minValue,
    required double maxValue,
  }) {
    final stats = _statsFor(sourceMetric);
    if (stats == null) {
      return _fallbackMetric(fallbackMetric, sourceMetric);
    }
    return _selectedFromData(
      stats: stats,
      fallbackMetric: fallbackMetric,
      sourceMetric: sourceMetric,
      selectionRule: 'blend_p90_upper_bound',
      dataSelected: stats.p90,
      minValue: minValue,
      maxValue: maxValue,
    );
  }

  ThresholdMetricProfile _buildLowerBound({
    required String sourceMetric,
    required ThresholdMetricProfile fallbackMetric,
    required double minValue,
    required double maxValue,
  }) {
    final stats = _statsFor(sourceMetric);
    if (stats == null) {
      return _fallbackMetric(fallbackMetric, sourceMetric);
    }
    return _selectedFromData(
      stats: stats,
      fallbackMetric: fallbackMetric,
      sourceMetric: sourceMetric,
      selectionRule: 'blend_p10_lower_bound',
      dataSelected: stats.p10,
      minValue: minValue,
      maxValue: maxValue,
    );
  }

  ThresholdMetricProfile _buildAngleAroundStraight({
    required String sourceMetric,
    required ThresholdMetricProfile fallbackMetric,
    required bool upperSide,
    required double minValue,
    required double maxValue,
  }) {
    final stats = _statsFor(sourceMetric);
    if (stats == null) {
      return _fallbackMetric(fallbackMetric, sourceMetric);
    }
    final fallbackDeviation = (180 - fallbackMetric.selected).abs();
    final allowedDeviation = max(fallbackDeviation, stats.p90);
    final dataSelected =
        upperSide ? 180 + allowedDeviation : 180 - allowedDeviation;
    return _selectedFromData(
      stats: stats,
      fallbackMetric: fallbackMetric,
      sourceMetric: sourceMetric,
      selectionRule: upperSide
          ? 'blend_straight_upper_deviation'
          : 'blend_straight_lower_deviation',
      dataSelected: dataSelected,
      minValue: minValue,
      maxValue: maxValue,
    );
  }

  ThresholdMetricProfile _selectedFromData({
    required _MetricStats stats,
    required ThresholdMetricProfile fallbackMetric,
    required String sourceMetric,
    required String selectionRule,
    required double dataSelected,
    required double minValue,
    required double maxValue,
  }) {
    final boundedData = _clampDouble(dataSelected, minValue, maxValue);
    final confidence = _confidence(stats.sampleCount);
    final selected = _clampDouble(
      _lerp(fallbackMetric.selected, boundedData, confidence),
      minValue,
      maxValue,
    );
    return ThresholdMetricProfile(
      selected: selected,
      p10: stats.p10,
      p50: stats.p50,
      p90: stats.p90,
      sampleCount: stats.sampleCount,
      selectionRule: selectionRule,
      sourceMetric: sourceMetric,
    );
  }

  ThresholdMetricProfile _fallbackMetric(
    ThresholdMetricProfile fallbackMetric,
    String sourceMetric,
  ) {
    return ThresholdMetricProfile(
      selected: fallbackMetric.selected,
      p10: fallbackMetric.p10,
      p50: fallbackMetric.p50,
      p90: fallbackMetric.p90,
      sampleCount: 0,
      selectionRule: fallbackMetric.selectionRule ?? 'fallback_default',
      sourceMetric: sourceMetric,
    );
  }

  _MetricStats? _statsFor(String sourceMetric) {
    final values = metricValues[sourceMetric];
    if (values == null || values.isEmpty) {
      return null;
    }
    final sorted = <double>[...values]..sort();
    return _MetricStats(
      sampleCount: sorted.length,
      p10: _quantile(sorted, 0.10),
      p50: _quantile(sorted, 0.50),
      p90: _quantile(sorted, 0.90),
    );
  }
}

class _MetricStats {
  const _MetricStats({
    required this.sampleCount,
    required this.p10,
    required this.p50,
    required this.p90,
  });

  final int sampleCount;
  final double p10;
  final double p50;
  final double p90;
}

class _BucketKey {
  const _BucketKey({
    required this.exercise,
    required this.view,
    required this.group,
  });

  final String exercise;
  final String view;
  final String group;

  @override
  bool operator ==(Object other) {
    return other is _BucketKey &&
        other.exercise == exercise &&
        other.view == view &&
        other.group == group;
  }

  @override
  int get hashCode => Object.hash(exercise, view, group);
}

class _IngestionStats {
  _IngestionStats({required this.path});

  final String path;
  bool missing = false;
  int totalLines = 0;
  int usableSamples = 0;
  int standardSamples = 0;
  int errorSamples = 0;
  int invalidJson = 0;
  int unsupportedExercise = 0;
  int skippedNoLandmarks = 0;

  String summaryLine() {
    if (missing) {
      return '[skip] $path not found';
    }
    return '[ingest] $path lines=$totalLines usable=$usableSamples '
        'standard=$standardSamples error=$errorSamples '
        'no_landmarks=$skippedNoLandmarks invalid_json=$invalidJson '
        'unsupported_exercise=$unsupportedExercise';
  }
}

Map<String, dynamic>? _decodeJsonObject(String line) {
  try {
    final decoded = jsonDecode(line);
    if (decoded is Map<String, dynamic>) {
      return decoded;
    }
  } catch (_) {
    return null;
  }
  return null;
}

Pose? _buildPose(Map<String, dynamic> record) {
  final landmarksJson = record['landmarks_final'];
  if (landmarksJson is! Map) {
    return null;
  }

  final landmarks = <PoseLandmarkType, PoseLandmark>{};
  for (final entry in landmarksJson.entries) {
    final type = _landmarkTypeByName(entry.key.toString());
    if (type == null || entry.value is! Map) {
      continue;
    }
    final value = Map<String, dynamic>.from(entry.value as Map);
    final x = (value['x'] as num?)?.toDouble();
    final y = (value['y'] as num?)?.toDouble();
    final z = (value['z'] as num?)?.toDouble() ?? 0.0;
    final likelihood = (value['likelihood'] as num?)?.toDouble();
    if (x == null || y == null || likelihood == null) {
      continue;
    }
    landmarks[type] = PoseLandmark(
      x: x,
      y: y,
      z: z,
      likelihood: likelihood,
    );
  }

  if (landmarks.isEmpty) {
    return null;
  }

  final timestampIso =
      record['capture_time_left'] as String? ?? record['timestamp'] as String?;
  final timestamp = DateTime.tryParse(timestampIso ?? '') ?? DateTime.now();
  return Pose(
    landmarks: landmarks,
    timestamp: timestamp,
    source: _sourceName(record, 'dataset'),
    depthMode: PoseDepthMode.monocular,
  );
}

PoseLandmarkType? _landmarkTypeByName(String rawName) {
  for (final type in PoseLandmarkType.values) {
    if (type.name == rawName) {
      return type;
    }
  }
  return null;
}

String? _normalizeExercise(String? exercise) {
  final normalized = exercise?.trim().toLowerCase();
  switch (normalized) {
    case 'squat':
    case 'pushup':
    case 'plank':
      return normalized;
    case 'push-up':
      return 'pushup';
    default:
      return null;
  }
}

String? _normalizeView(String? view) {
  final normalized = view?.trim().toLowerCase();
  if (normalized == null || normalized.isEmpty || normalized == 'unknown') {
    return null;
  }
  if (normalized.contains('front')) {
    return 'front';
  }
  if (normalized.contains('side') || normalized.contains('profile')) {
    return 'side';
  }
  if (normalized.contains('oblique') || normalized.contains('45')) {
    return 'oblique';
  }
  return null;
}

String? _normalizeGender(String? gender) {
  final normalized = gender?.trim().toLowerCase();
  switch (normalized) {
    case 'male':
    case 'man':
      return 'male';
    case 'female':
    case 'woman':
      return 'female';
    default:
      return null;
  }
}

String? _readGender(Map<String, dynamic> record) {
  final userProfile = _userProfile(record);
  return userProfile['gender'] as String? ?? record['gender'] as String?;
}

double? _readHeightCm(Map<String, dynamic> record) {
  final userProfile = _userProfile(record);
  return (userProfile['heightCm'] as num?)?.toDouble() ??
      (record['heightCm'] as num?)?.toDouble();
}

Map<String, dynamic> _userProfile(Map<String, dynamic> record) {
  final userProfile = record['user_profile'];
  if (userProfile is Map<String, dynamic>) {
    return userProfile;
  }
  if (userProfile is Map) {
    return Map<String, dynamic>.from(userProfile);
  }
  return const <String, dynamic>{};
}

String? _heightGroup(double? heightCm) {
  if (heightCm == null) {
    return null;
  }
  if (heightCm < 160) {
    return 'short';
  }
  if (heightCm > 180) {
    return 'tall';
  }
  return 'medium';
}

List<String> _groupsFor({
  required String? gender,
  required String? heightGroup,
}) {
  final groups = <String>{'default'};
  if (gender != null) {
    groups.add(gender);
  }
  if (heightGroup != null) {
    groups.add(heightGroup);
  }
  if (gender != null && heightGroup != null) {
    groups.add('${gender}_$heightGroup');
  }
  return groups.toList();
}

String _sourceName(Map<String, dynamic> record, String fallback) {
  return record['source_dataset'] as String? ??
      record['source'] as String? ??
      fallback;
}

List<String> _poseErrors(Map<String, dynamic> record) {
  final poseErrors = record['pose_error_types'];
  if (poseErrors is List) {
    return poseErrors
        .map((item) => item.toString())
        .where((item) => item.isNotEmpty)
        .toList();
  }
  return const <String>[];
}

double _quantile(List<double> sortedValues, double percentile) {
  if (sortedValues.isEmpty) {
    return 0.0;
  }
  if (sortedValues.length == 1) {
    return sortedValues.first;
  }
  final clamped = percentile.clamp(0.0, 1.0);
  final position = (sortedValues.length - 1) * clamped;
  final lower = position.floor();
  final upper = position.ceil();
  if (lower == upper) {
    return sortedValues[lower];
  }
  final mix = position - lower;
  return _lerp(sortedValues[lower], sortedValues[upper], mix);
}

double _confidence(int sampleCount) {
  return (sampleCount / 20.0).clamp(0.0, 1.0).toDouble();
}

double _lerp(double start, double end, double t) {
  return start + (end - start) * t;
}

double _clampDouble(num value, double minValue, double maxValue) {
  return value.clamp(minValue, maxValue).toDouble();
}

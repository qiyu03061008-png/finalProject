import 'dart:math' as math;

import '../models/analysis_result.dart';
import '../models/pose_landmark.dart';
import '../models/user_profile.dart';
import 'pose_metric_calculator.dart';

class PoseAnalyzer {
  static const int _stateHoldFrames = 2;
  static const int _maxMissingFramesToKeepState = 6;
  static const int _minRepPhaseMs = 140;
  static const String _cleanRepPraise = '动作不错。继续保持';

  static const int _issueOnFrames = 2;
  static const int _issueOffFrames = 1;
  static const double _emaFast = 0.45;
  static const double _emaSlow = 0.35;

  final PoseMetricCalculator _metricCalculator = const PoseMetricCalculator();

  int _squatCount = 0;
  int _pushupCount = 0;
  double _plankHoldSeconds = 0;

  bool _squatDown = false;
  bool _pushupDown = false;
  int _squatDownStreak = 0;
  int _squatUpStreak = 0;
  int _pushupDownStreak = 0;
  int _pushupUpStreak = 0;
  int _squatMissingFrames = 0;
  int _pushupMissingFrames = 0;

  bool _squatRepSeenDown = false;
  bool _squatRepReachedDepth = false;
  bool _squatRepHasIssue = false;
  double? _squatRepMinAngle;
  DateTime? _squatPhaseChangedAt;

  bool _pushupRepSeenDown = false;
  bool _pushupRepReachedDepth = false;
  bool _pushupRepHasIssue = false;
  double? _pushupRepMinAngle;
  DateTime? _pushupPhaseChangedAt;

  DateTime? _lastPlankTimestamp;

  final Map<String, double> _emaMetrics = <String, double>{};
  final Map<PoseErrorType, int> _issueOnStreak = <PoseErrorType, int>{};
  final Map<PoseErrorType, int> _issueOffStreak = <PoseErrorType, int>{};
  final Set<PoseErrorType> _latchedIssues = <PoseErrorType>{};

  ExerciseAnalysisResult analyze({
    required Pose pose,
    required ExerciseType exerciseType,
    required UserProfile profile,
    String? viewTag,
  }) {
    final snapshot = _metricCalculator.calculate(pose);
    final resolvedViewTag =
        _normalizeViewTag(viewTag) ?? snapshot.inferredViewTag;
    final thresholds = PersonalizedThresholds.fromProfile(
      profile,
      viewTag: resolvedViewTag,
    );

    switch (exerciseType) {
      case ExerciseType.squat:
        return _analyzeSquat(pose, snapshot, thresholds, profile);
      case ExerciseType.pushup:
        return _analyzePushup(pose, snapshot, thresholds, profile);
      case ExerciseType.plank:
        return _analyzePlank(pose, snapshot, thresholds, profile);
    }
  }

  ExerciseAnalysisResult _analyzeSquat(
    Pose pose,
    PoseMetricSnapshot snapshot,
    PersonalizedThresholds t,
    UserProfile profile,
  ) {
    final issues = <PoseIssue>[];
    var countDelta = 0;
    var repJustCounted = false;
    var repJustCountedClean = false;

    final isMoveNet = pose.source.startsWith('movenet');
    final holdFrames =
        (pose.source == 'blazepose' || isMoveNet) ? 1 : _stateHoldFrames;
    final squatUpGate = isMoveNet
        ? (t.squatUpAngle - 20)
        : pose.source == 'blazepose'
            ? (t.squatUpAngle - 10)
            : (t.squatUpAngle - 6);
    final depthReachFactor = isMoveNet ? 1.05 : 0.65;

    final kneeAngle = _smoothMetric(
      'squat_knee_angle',
      snapshot['squatKneeAngle'],
      alpha: _emaFast,
    );
    final torsoLeanDeg = _smoothMetric(
      'squat_torso_lean',
      snapshot['squatTorsoLeanDeg'],
      alpha: _emaSlow,
    );
    final kneeInward = _smoothMetric(
      'squat_knee_inward_ratio',
      snapshot['squatKneeInwardRatio'],
      alpha: _emaSlow,
    );
    final kneeOverToe = _smoothMetric(
      'squat_knee_over_toe',
      snapshot['squatKneeOverToe'],
      alpha: _emaSlow,
    );

    if (kneeAngle != null) {
      _squatMissingFrames = 0;

      if (kneeAngle < t.squatDownAngle) {
        _squatDownStreak += 1;
        _squatUpStreak = 0;
        if (_squatDownStreak >= holdFrames && !_squatDown) {
          _squatDown = true;
          _squatRepSeenDown = true;
          _squatRepMinAngle = kneeAngle;
          _squatPhaseChangedAt = pose.timestamp;
        }
      } else if (kneeAngle > squatUpGate) {
        _squatUpStreak += 1;
        _squatDownStreak = 0;
        if (_squatUpStreak >= holdFrames && _squatDown) {
          final phaseDurationMs = _squatPhaseChangedAt == null
              ? _minRepPhaseMs
              : pose.timestamp.difference(_squatPhaseChangedAt!).inMilliseconds;
          _squatDown = false;
          _squatPhaseChangedAt = pose.timestamp;
          if (_squatRepSeenDown &&
              _squatRepReachedDepth &&
              phaseDurationMs >= _minRepPhaseMs &&
              _hasSquatRecoveredEnough(kneeAngle, t, isMoveNet)) {
            _squatCount += 1;
            countDelta = 1;
            repJustCounted = true;
            repJustCountedClean = !_squatRepHasIssue;
          }
          _squatRepSeenDown = false;
          _squatRepReachedDepth = false;
          _squatRepHasIssue = false;
          _squatRepMinAngle = null;
        }
      } else {
        _squatDownStreak = 0;
        _squatUpStreak = 0;
      }

      if (_squatDown) {
        _squatRepMinAngle = _squatRepMinAngle == null
            ? kneeAngle
            : math.min(_squatRepMinAngle!, kneeAngle);
      }

      if (_squatDown &&
          kneeAngle <=
              (t.squatDownAngle + t.shallowSquatMargin * depthReachFactor)) {
        _squatRepReachedDepth = true;
      }

      final shallow = _stableIssue(
        type: PoseErrorType.shallowSquat,
        active:
            kneeAngle > t.squatDownAngle + t.shallowSquatMargin && _squatDown,
        onFrames: isMoveNet ? 1 : _issueOnFrames,
        build: () => const PoseIssue(
          type: PoseErrorType.shallowSquat,
          message: '下蹲深度不够',
          suggestion: '继续下蹲，稳定达到目标深度后再起身',
          severity: 0.6,
        ),
      );
      if (shallow != null) {
        issues.add(shallow);
        _squatRepHasIssue = true;
      }
    } else {
      _squatMissingFrames += 1;
      _squatDownStreak = 0;
      _squatUpStreak = 0;
      if (_squatMissingFrames > _maxMissingFramesToKeepState) {
        _squatDown = false;
        _squatRepSeenDown = false;
        _squatRepReachedDepth = false;
        _squatRepHasIssue = false;
        _squatRepMinAngle = null;
        _squatPhaseChangedAt = null;
      }
    }

    if (torsoLeanDeg != null) {
      final torsoLeanIssue = _stableIssue(
        type: PoseErrorType.torsoLeanForward,
        active: torsoLeanDeg > t.maxTorsoLeanDeg - (isMoveNet ? 2 : 0),
        build: () => PoseIssue(
          type: PoseErrorType.torsoLeanForward,
          message: '躯干前倾过多',
          suggestion:
              '抬胸并收紧核心。当前前倾 ${torsoLeanDeg.toStringAsFixed(1)}°，建议不超过 ${t.maxTorsoLeanDeg.toStringAsFixed(1)}°',
          severity: 0.72,
        ),
      );
      if (torsoLeanIssue != null) {
        issues.add(torsoLeanIssue);
        _squatRepHasIssue = true;
      }
    }

    if (kneeInward != null) {
      final stanceHint = profile.shoulderToHipRatio > 1.25
          ? '略微加宽站距，并主动把膝盖向外打开'
          : '让膝盖始终跟随脚尖方向移动';
      final kneeValgusIssue = _stableIssue(
        type: PoseErrorType.kneeValgus,
        active: kneeInward < t.maxKneeInwardRatio + (isMoveNet ? 0.03 : 0),
        build: () => PoseIssue(
          type: PoseErrorType.kneeValgus,
          message: '检测到膝盖内扣',
          suggestion:
              '$stanceHint。当前比值 ${kneeInward.toStringAsFixed(2)}，建议不低于 ${t.maxKneeInwardRatio.toStringAsFixed(2)}',
          severity: 0.82,
        ),
      );
      if (kneeValgusIssue != null) {
        issues.add(kneeValgusIssue);
        _squatRepHasIssue = true;
      }
    }

    if (kneeOverToe != null) {
      final limbHint = profile.legLengthRatio > 0.56
          ? '先向后坐髋，再让膝盖自然前移'
          : '稳住脚跟，控制膝盖前移幅度';
      final kneeToeIssue = _stableIssue(
        type: PoseErrorType.kneeOverToe,
        active: kneeOverToe > t.maxKneeOverToe - (isMoveNet ? 0.03 : 0),
        build: () => PoseIssue(
          type: PoseErrorType.kneeOverToe,
          message: '膝盖前移过多',
          suggestion:
              '$limbHint。当前值 ${kneeOverToe.toStringAsFixed(2)}，建议不高于 ${t.maxKneeOverToe.toStringAsFixed(2)}',
          severity: 0.55,
        ),
      );
      if (kneeToeIssue != null) {
        issues.add(kneeToeIssue);
        _squatRepHasIssue = true;
      }
    }

    final sortedIssues = _sortedIssues(issues);
    final score = _scoreFromIssues(sortedIssues);
    final feedback = _feedbackFromIssues(
      sortedIssues,
      fallback: repJustCountedClean ? _cleanRepPraise : '请先完成一个标准动作',
    );

    return ExerciseAnalysisResult(
      feedback: feedback,
      score: score,
      count: _squatCount,
      countDelta: countDelta,
      repJustCounted: repJustCounted,
      repJustCountedClean: repJustCountedClean,
      issues: sortedIssues,
      depthModeLabel: '单目 3D',
      metrics: <String, double>{
        'knee_angle': kneeAngle ?? 0.0,
        'torso_lean_deg': torsoLeanDeg ?? 0.0,
        'knee_inward_ratio': kneeInward ?? 1.0,
        'knee_over_toe': kneeOverToe ?? 0.0,
        'height_cm': profile.heightCm,
      },
    );
  }

  ExerciseAnalysisResult _analyzePushup(
    Pose pose,
    PoseMetricSnapshot snapshot,
    PersonalizedThresholds t,
    UserProfile profile,
  ) {
    final issues = <PoseIssue>[];
    var countDelta = 0;
    var repJustCounted = false;
    var repJustCountedClean = false;

    final isMoveNet = pose.source.startsWith('movenet');
    final holdFrames =
        (pose.source == 'blazepose' || isMoveNet) ? 1 : _stateHoldFrames;
    final pushupUpGate = isMoveNet
        ? (t.pushupUpAngle - 14)
        : pose.source == 'blazepose'
            ? (t.pushupUpAngle - 10)
            : (t.pushupUpAngle - 6);

    final elbowAngle = _smoothMetric(
      'pushup_elbow_angle',
      snapshot['pushupElbowAngle'],
      alpha: _emaFast,
    );
    final bodyLineAngle = _smoothMetric(
      'pushup_body_line_angle',
      snapshot['pushupBodyLineAngle'],
      alpha: _emaFast,
    );
    final bodyLineDeviation = _smoothMetric(
      'pushup_body_line_deviation',
      snapshot['pushupBodyLineDeviation'],
      alpha: _emaSlow,
    );
    final hipOffset = _smoothMetric(
      'pushup_hip_offset',
      snapshot['pushupHipOffset'],
      alpha: _emaSlow,
    );
    final elbowFlare = _smoothMetric(
      'pushup_elbow_flare',
      snapshot['pushupElbowFlareDeg'],
      alpha: _emaSlow,
    );

    if (elbowAngle != null) {
      _pushupMissingFrames = 0;

      if (elbowAngle < t.pushupDownAngle) {
        _pushupDownStreak += 1;
        _pushupUpStreak = 0;
        if (_pushupDownStreak >= holdFrames && !_pushupDown) {
          _pushupDown = true;
          _pushupRepSeenDown = true;
          _pushupRepMinAngle = elbowAngle;
          _pushupPhaseChangedAt = pose.timestamp;
        }
      } else if (elbowAngle > pushupUpGate) {
        _pushupUpStreak += 1;
        _pushupDownStreak = 0;
        if (_pushupUpStreak >= holdFrames && _pushupDown) {
          final phaseDurationMs = _pushupPhaseChangedAt == null
              ? _minRepPhaseMs
              : pose.timestamp
                  .difference(_pushupPhaseChangedAt!)
                  .inMilliseconds;
          _pushupDown = false;
          _pushupPhaseChangedAt = pose.timestamp;
          if (_pushupRepSeenDown &&
              _pushupRepReachedDepth &&
              phaseDurationMs >= _minRepPhaseMs &&
              _hasPushupRecoveredEnough(elbowAngle, t, isMoveNet)) {
            _pushupCount += 1;
            countDelta = 1;
            repJustCounted = true;
            repJustCountedClean = !_pushupRepHasIssue;
          }
          _pushupRepSeenDown = false;
          _pushupRepReachedDepth = false;
          _pushupRepHasIssue = false;
          _pushupRepMinAngle = null;
        }
      } else {
        _pushupDownStreak = 0;
        _pushupUpStreak = 0;
      }

      if (_pushupDown) {
        _pushupRepMinAngle = _pushupRepMinAngle == null
            ? elbowAngle
            : math.min(_pushupRepMinAngle!, elbowAngle);
      }

      if (_pushupDown &&
          elbowAngle <= (t.pushupDownAngle + t.pushupDepthMargin * 0.65)) {
        _pushupRepReachedDepth = true;
      }

      final depthIssue = _stableIssue(
        type: PoseErrorType.pushupDepthNotEnough,
        active:
            elbowAngle > t.pushupDownAngle + t.pushupDepthMargin && _pushupDown,
        build: () => PoseIssue(
          type: PoseErrorType.pushupDepthNotEnough,
          message: '俯卧撑下放深度不够',
          suggestion:
              '继续下放，肘角尽量接近 ${t.pushupDownAngle.toStringAsFixed(0)}° 后再推起',
          severity: 0.6,
        ),
      );
      if (depthIssue != null) {
        issues.add(depthIssue);
        _pushupRepHasIssue = true;
      }
    } else {
      _pushupMissingFrames += 1;
      _pushupDownStreak = 0;
      _pushupUpStreak = 0;
      if (_pushupMissingFrames > _maxMissingFramesToKeepState) {
        _pushupDown = false;
        _pushupRepSeenDown = false;
        _pushupRepReachedDepth = false;
        _pushupRepHasIssue = false;
        _pushupRepMinAngle = null;
        _pushupPhaseChangedAt = null;
      }
    }

    if (hipOffset != null && bodyLineDeviation != null) {
      final sagIssue = _stableIssue(
        type: PoseErrorType.pushupHipSag,
        active: hipOffset < t.pushupHipSagOffset &&
            bodyLineDeviation > (180 - t.maxHipSagDeg).abs(),
        build: () {
          final braceHint = profile.weightKg > 80
              ? '放慢速度，每次动作前先收紧核心'
              : '下放前先收紧核心和臀部';
          return PoseIssue(
            type: PoseErrorType.pushupHipSag,
            message: '检测到塌腰',
            suggestion: '$braceHint，尽量保持肩、髋、踝接近一条直线',
            severity: 0.8,
          );
        },
      );
      if (sagIssue != null) {
        issues.add(sagIssue);
        _pushupRepHasIssue = true;
      }

      final pikeIssue = _stableIssue(
        type: PoseErrorType.pushupHipPike,
        active: hipOffset > t.pushupHipPikeOffset &&
            bodyLineDeviation > (t.maxHipPikeDeg - 180).abs(),
        build: () {
          final armHint = profile.armLengthRatio > 0.47
              ? '可以轻微前倾，但仍需要主动把臀部放低'
              : '把臀部略微放低，保持肩膀和骨盆更好对齐';
          return PoseIssue(
            type: PoseErrorType.pushupHipPike,
            message: '检测到撅臀',
            suggestion: armHint,
            severity: 0.7,
          );
        },
      );
      if (pikeIssue != null) {
        issues.add(pikeIssue);
        _pushupRepHasIssue = true;
      }
    }

    if (elbowFlare != null) {
      final flareIssue = _stableIssue(
        type: PoseErrorType.pushupElbowFlare,
        active: elbowFlare > t.maxElbowFlareDeg - (isMoveNet ? 4 : 0),
        build: () => PoseIssue(
          type: PoseErrorType.pushupElbowFlare,
          message: '手肘外展过大',
          suggestion:
              '手肘尽量保持在约 45° 到 60°。当前 ${elbowFlare.toStringAsFixed(0)}°，建议不高于 ${t.maxElbowFlareDeg.toStringAsFixed(0)}°',
          severity: 0.55,
        ),
      );
      if (flareIssue != null) {
        issues.add(flareIssue);
        _pushupRepHasIssue = true;
      }
    }

    final sortedIssues = _sortedIssues(issues);
    final score = _scoreFromIssues(sortedIssues);
    final feedback = _feedbackFromIssues(
      sortedIssues,
      fallback: repJustCountedClean ? _cleanRepPraise : '请先完成一个标准动作',
    );

    return ExerciseAnalysisResult(
      feedback: feedback,
      score: score,
      count: _pushupCount,
      countDelta: countDelta,
      repJustCounted: repJustCounted,
      repJustCountedClean: repJustCountedClean,
      issues: sortedIssues,
      depthModeLabel: '单目 3D',
      metrics: <String, double>{
        'elbow_angle': elbowAngle ?? 0.0,
        'body_line_angle': bodyLineAngle ?? 0.0,
        'body_line_deviation': bodyLineDeviation ?? 0.0,
        'hip_offset_ratio': hipOffset ?? 0.0,
        'elbow_flare_deg': elbowFlare ?? 0.0,
        'weight_kg': profile.weightKg,
      },
    );
  }

  ExerciseAnalysisResult _analyzePlank(
    Pose pose,
    PoseMetricSnapshot snapshot,
    PersonalizedThresholds t,
    UserProfile profile,
  ) {
    final issues = <PoseIssue>[];

    final bodyLineAngle = _smoothMetric(
      'plank_body_line_angle',
      snapshot['plankBodyLineAngle'],
      alpha: _emaFast,
    );
    final bodyLineDeviation = _smoothMetric(
      'plank_body_line_deviation',
      snapshot['plankBodyLineDeviation'],
      alpha: _emaSlow,
    );
    final hipOffset = _smoothMetric(
      'plank_hip_offset',
      snapshot['plankHipOffset'],
      alpha: _emaSlow,
    );
    final neckAngle = _smoothMetric(
      'plank_neck_angle',
      snapshot['plankNeckAngle'],
      alpha: _emaSlow,
    );

    final now = pose.timestamp;
    final dt = _lastPlankTimestamp == null
        ? 0.0
        : now.difference(_lastPlankTimestamp!).inMilliseconds / 1000.0;
    _lastPlankTimestamp = now;

    final neutralDeviationLimit = math.max(
      (180 - t.plankNeutralMin).abs(),
      (t.plankNeutralMax - 180).abs(),
    );

    if (hipOffset != null && bodyLineDeviation != null) {
      final sagIssue = _stableIssue(
        type: PoseErrorType.plankHipSag,
        active: hipOffset < t.plankHipSagOffset &&
            bodyLineDeviation > neutralDeviationLimit,
        build: () => const PoseIssue(
          type: PoseErrorType.plankHipSag,
          message: '平板支撑出现塌腰',
          suggestion: '收紧腹部和臀部，并轻微后倾骨盆',
          severity: 0.8,
        ),
      );
      if (sagIssue != null) {
        issues.add(sagIssue);
      }

      final pikeIssue = _stableIssue(
        type: PoseErrorType.plankHipPike,
        active: hipOffset > t.plankHipPikeOffset &&
            bodyLineDeviation > neutralDeviationLimit,
        build: () => const PoseIssue(
          type: PoseErrorType.plankHipPike,
          message: '平板支撑出现撅臀',
          suggestion: '适当降低臀部，保持肩到踝基本对齐',
          severity: 0.7,
        ),
      );
      if (pikeIssue != null) {
        issues.add(pikeIssue);
      }
    }

    if (neckAngle != null) {
      final neckIssue = _stableIssue(
        type: PoseErrorType.plankNeckNotNeutral,
        active: neckAngle < t.maxNeckAngle,
        build: () => PoseIssue(
          type: PoseErrorType.plankNeckNotNeutral,
          message: '颈部姿态不够中立',
          suggestion:
              '视线看向前下方，尽量保持耳、肩、髋接近同一平面。当前 ${neckAngle.toStringAsFixed(0)}°，建议不低于 ${t.maxNeckAngle.toStringAsFixed(0)}°',
          severity: 0.45,
        ),
      );
      if (neckIssue != null) {
        issues.add(neckIssue);
      }
    }

    if (issues.isEmpty && dt > 0 && dt < 0.8) {
      _plankHoldSeconds += dt;
    }

    final sortedIssues = _sortedIssues(issues);
    final score = _scoreFromIssues(sortedIssues);
    final feedback = _feedbackFromIssues(
      sortedIssues,
      fallback: '平板支撑稳定，继续保持',
    );

    return ExerciseAnalysisResult(
      feedback: feedback,
      score: score,
      count: _plankHoldSeconds.floor(),
      countDelta: 0,
      repJustCounted: false,
      repJustCountedClean: false,
      issues: sortedIssues,
      depthModeLabel: '单目 3D',
      metrics: <String, double>{
        'body_line_angle': bodyLineAngle ?? 0.0,
        'body_line_deviation': bodyLineDeviation ?? 0.0,
        'hip_offset_ratio': hipOffset ?? 0.0,
        'neck_angle': neckAngle ?? 0.0,
        'hold_seconds': _plankHoldSeconds,
        'height_cm': profile.heightCm,
      },
    );
  }

  void reset(ExerciseType type) {
    switch (type) {
      case ExerciseType.squat:
        _squatCount = 0;
        _squatDown = false;
        _squatDownStreak = 0;
        _squatUpStreak = 0;
        _squatMissingFrames = 0;
        _squatRepSeenDown = false;
        _squatRepReachedDepth = false;
        _squatRepHasIssue = false;
        _squatRepMinAngle = null;
        _squatPhaseChangedAt = null;
        break;
      case ExerciseType.pushup:
        _pushupCount = 0;
        _pushupDown = false;
        _pushupDownStreak = 0;
        _pushupUpStreak = 0;
        _pushupMissingFrames = 0;
        _pushupRepSeenDown = false;
        _pushupRepReachedDepth = false;
        _pushupRepHasIssue = false;
        _pushupRepMinAngle = null;
        _pushupPhaseChangedAt = null;
        break;
      case ExerciseType.plank:
        _plankHoldSeconds = 0;
        _lastPlankTimestamp = null;
        break;
    }
    _emaMetrics.clear();
    _issueOnStreak.clear();
    _issueOffStreak.clear();
    _latchedIssues.clear();
  }

  int _scoreFromIssues(List<PoseIssue> issues) {
    final deduction = issues.fold<double>(
      0,
      (sum, item) => sum + item.severity * 22,
    );
    return (100 - deduction).clamp(35, 100).round();
  }

  List<PoseIssue> _sortedIssues(List<PoseIssue> issues) {
    final sorted = <PoseIssue>[...issues]
      ..sort((a, b) => b.severity.compareTo(a.severity));
    return sorted;
  }

  String _feedbackFromIssues(List<PoseIssue> issues, {required String fallback}) {
    if (issues.isEmpty) {
      return fallback;
    }
    final sorted = <PoseIssue>[...issues]
      ..sort((a, b) => b.severity.compareTo(a.severity));
    return sorted.first.message;
  }

  PoseIssue? _stableIssue({
    required PoseErrorType type,
    required bool active,
    required PoseIssue Function() build,
    int onFrames = _issueOnFrames,
    int offFrames = _issueOffFrames,
  }) {
    if (active) {
      _issueOnStreak[type] = (_issueOnStreak[type] ?? 0) + 1;
      _issueOffStreak[type] = 0;
      if ((_issueOnStreak[type] ?? 0) >= onFrames) {
        _latchedIssues.add(type);
      }
    } else {
      _issueOnStreak[type] = 0;
      _issueOffStreak[type] = (_issueOffStreak[type] ?? 0) + 1;
      if ((_issueOffStreak[type] ?? 0) >= offFrames) {
        _latchedIssues.remove(type);
      }
    }
    return _latchedIssues.contains(type) ? build() : null;
  }

  double? _smoothMetric(String key, double? value, {double alpha = _emaFast}) {
    if (value == null || value.isNaN || value.isInfinite) {
      return null;
    }
    final previous = _emaMetrics[key];
    final smoothed =
        previous == null ? value : previous * (1 - alpha) + value * alpha;
    _emaMetrics[key] = smoothed;
    return smoothed;
  }

  bool _hasSquatRecoveredEnough(
    double kneeAngle,
    PersonalizedThresholds t,
    bool isMoveNet,
  ) {
    final minAngle = _squatRepMinAngle;
    final nearTop = kneeAngle >= (t.squatUpAngle - (isMoveNet ? 24 : 16));
    final clearRebound = minAngle != null &&
        kneeAngle >=
            math.max(
              minAngle + (isMoveNet ? 10 : 18),
              t.squatDownAngle + (isMoveNet ? 8 : 14),
            );
    return nearTop || clearRebound;
  }

  bool _hasPushupRecoveredEnough(
    double elbowAngle,
    PersonalizedThresholds t,
    bool isMoveNet,
  ) {
    final minAngle = _pushupRepMinAngle;
    final nearTop = elbowAngle >= (t.pushupUpAngle - (isMoveNet ? 18 : 16));
    final clearRebound = minAngle != null &&
        elbowAngle >=
            math.max(
              minAngle + (isMoveNet ? 18 : 22),
              t.pushupDownAngle + (isMoveNet ? 14 : 16),
            );
    return nearTop || clearRebound;
  }

  String? _normalizeViewTag(String? viewTag) {
    final normalized = viewTag?.trim().toLowerCase();
    switch (normalized) {
      case 'front':
      case 'side':
      case 'oblique':
        return normalized;
      default:
        return null;
    }
  }
}

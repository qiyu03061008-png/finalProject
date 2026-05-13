import 'dart:math' as math;

import '../models/analysis_result.dart';
import '../models/pose_landmark.dart';
import '../models/user_profile.dart';
import 'pose_metric_calculator.dart';

class PoseAnalyzer {
  // 状态至少持续若干帧才算真正切换，避免单帧抖动导致误判。
  static const int _stateHoldFrames = 2;
  static const int _maxMissingFramesToKeepState = 6;
  // 一次完整动作至少要有一个最短时长，过滤“抖一下就计数”的情况。
  static const int _minRepPhaseMs = 420;
  static const int _moveNetStateHoldFrames = 3;
  static const int _blazePoseStateHoldFrames = 2;
  static const String _cleanRepPraise = '动作不错。继续保持';

  static const int _issueOnFrames = 2;
  static const int _issueOffFrames = 2;
  static const double _emaFast = 0.32;
  static const double _emaSlow = 0.24;

  final PoseMetricCalculator _metricCalculator = const PoseMetricCalculator();

  int _squatCount = 0;
  int _pushupCount = 0;
  double _plankHoldSeconds = 0;

  // `Down` 表示当前是否处在动作最低点附近。
  bool _squatDown = false;
  bool _pushupDown = false;
  // streak 用连续帧数确认状态切换，而不是看单帧结果。
  int _squatDownStreak = 0;
  int _squatUpStreak = 0;
  int _pushupDownStreak = 0;
  int _pushupUpStreak = 0;
  int _squatMissingFrames = 0;
  int _pushupMissingFrames = 0;

  // 下面这一组状态用来描述“这一整次动作”是否合格。
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

  /// 分析单帧姿态数据，并按动作类型分发到对应的分析逻辑。
  ExerciseAnalysisResult analyze({
    required Pose pose,
    required ExerciseType exerciseType,
    required UserProfile profile,
    String? viewTag,
  }) {
    final snapshot = _metricCalculator.calculate(pose);
    // 外部传入的视角优先；如果没有，就使用当前帧自动推断的视角。
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

  /// 分析深蹲动作，判断状态切换、计数结果和常见动作问题。
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
    // 不同模型稳定性不同，所以进入/离开动作相位的门槛会略有区别。
    final holdFrames = isMoveNet
        ? _moveNetStateHoldFrames
        : pose.source == 'blazepose'
        ? _blazePoseStateHoldFrames
        : _stateHoldFrames;
    final squatUpGate = isMoveNet
        ? (t.squatUpAngle - 20)
        : pose.source == 'blazepose'
        ? (t.squatUpAngle - 10)
        : (t.squatUpAngle - 6);

// 进入“正在下蹲”的角度要比真正达标角度宽松。
// 否则浅蹲动作根本进不了 _squatDown，自然不会提示浅蹲。
    final squatStartAngle = math.min(
      t.squatUpAngle - 8,
      t.squatDownAngle + 34,
    );

    final depthReachFactor = isMoveNet ? 0.60 : 0.65;

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

      // 膝角足够小，说明正在下蹲。
      if (kneeAngle < squatStartAngle)  {
        _squatDownStreak += 1;
        _squatUpStreak = 0;
        if (_squatDownStreak >= holdFrames && !_squatDown) {
          // 真正进入下蹲相位时，记录这一轮动作已经“下去过”。
          _squatDown = true;
          _squatRepSeenDown = true;
          _squatRepMinAngle = kneeAngle;
          _squatPhaseChangedAt = pose.timestamp;
        }
      // 膝角重新变大，说明正在起身。
      } else if (kneeAngle > squatUpGate) {
        _squatUpStreak += 1;
        _squatDownStreak = 0;
        if (_squatUpStreak >= holdFrames && _squatDown) {
          final phaseDurationMs = _squatPhaseChangedAt == null
              ? _minRepPhaseMs
              : pose.timestamp.difference(_squatPhaseChangedAt!).inMilliseconds;
          _squatDown = false;
          _squatPhaseChangedAt = pose.timestamp;
          // 只有“下去过 + 深度够 + 速度不过快 + 起身够明显”才算一次完整动作。
          final squatMotionEnough = _squatRepMinAngle != null &&
              kneeAngle - _squatRepMinAngle! >= (isMoveNet ? 32 : 26);

          if (_squatRepSeenDown &&
              _squatRepReachedDepth &&
              squatMotionEnough &&
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

      // 在整个下蹲过程中，持续更新本次动作的最小膝角。
      if (_squatDown) {
        _squatRepMinAngle = _squatRepMinAngle == null
            ? kneeAngle
            : math.min(_squatRepMinAngle!, kneeAngle);
      }

      // 只要下蹲时达到过目标深度，就把本轮动作标记为“深度达标”。
      if (_squatDown &&
          kneeAngle <=
              (t.squatDownAngle + t.shallowSquatMargin * depthReachFactor)) {
        _squatRepReachedDepth = true;
      }

      final shallow = _stableIssue(
        type: PoseErrorType.shallowSquat,
        active: _squatDown &&
            !_squatRepReachedDepth &&
            _squatRepMinAngle != null &&
            kneeAngle > _squatRepMinAngle! + 6 &&
            kneeAngle > t.squatDownAngle + t.shallowSquatMargin * 0.8,
        onFrames: isMoveNet ? 1 : _issueOnFrames,
        build: () => const PoseIssue(
          type: PoseErrorType.shallowSquat,
          message: '下蹲深度不够',
          suggestion: '刚才下蹲没有达到目标深度，下次继续下蹲后再起身',
          severity: 0.68,
        ),
      );
      if (shallow != null) {
        issues.add(shallow);
        _squatRepHasIssue = true;
      }
    } else {
      // 关键角度暂时丢失时，不马上清空状态，给检测抖动留一点容错空间。
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
        active: (_squatDown || _squatRepSeenDown) &&
            torsoLeanDeg > t.maxTorsoLeanDeg + (isMoveNet ? 0 : 3),
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
        active: (_squatDown || _squatRepSeenDown) &&
            kneeInward < t.maxKneeInwardRatio + (isMoveNet ? 0.03 : 0),
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
        active: (_squatDown || _squatRepSeenDown) &&
            kneeOverToe > t.maxKneeOverToe - (isMoveNet ? 0.03 : 0),
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

  /// 分析俯卧撑动作，判断状态切换、计数结果和动作问题。
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
    if (!_isFloorBodyPose(pose)) {
      _pushupDown = false;
      _pushupDownStreak = 0;
      _pushupUpStreak = 0;
      _pushupRepSeenDown = false;
      _pushupRepReachedDepth = false;
      _pushupRepHasIssue = false;
      _pushupRepMinAngle = null;
      _pushupPhaseChangedAt = null;

      return ExerciseAnalysisResult(
        feedback: '请先摆出俯卧撑准备姿势',
        score: 0,
        count: _pushupCount,
        countDelta: 0,
        repJustCounted: false,
        repJustCountedClean: false,
        issues: const <PoseIssue>[],
        depthModeLabel: '单目 3D',
        metrics: const <String, double>{},
      );
    }
    // 与深蹲相同，俯卧撑也会按模型来源微调判定门槛。
    final holdFrames = isMoveNet
        ? _moveNetStateHoldFrames
        : pose.source == 'blazepose'
        ? _blazePoseStateHoldFrames
        : _stateHoldFrames;
    final pushupUpGate = isMoveNet
        ? (t.pushupUpAngle - 14)
        : pose.source == 'blazepose'
            ? (t.pushupUpAngle - 10)
            : (t.pushupUpAngle - 6);
    final pushupStartAngle = math.min(
      t.pushupUpAngle - 10,
      t.pushupDownAngle + t.pushupDepthMargin + 28,
    );

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

      // 手肘弯曲到足够小，说明已经下放到底部附近。
      if (elbowAngle < pushupStartAngle) {
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
          // 俯卧撑计数逻辑与深蹲一致：必须完成一整次“下去再起来”。
          final pushupMotionEnough = _pushupRepMinAngle != null &&
              elbowAngle - _pushupRepMinAngle! >= (isMoveNet ? 42 : 34);

          if (_pushupRepSeenDown &&
              _pushupRepReachedDepth &&
              pushupMotionEnough &&
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

      // 记录这一轮俯卧撑最低时的手肘角度，后面用来判断是否真的推起。
      if (_pushupDown) {
        _pushupRepMinAngle = _pushupRepMinAngle == null
            ? elbowAngle
            : math.min(_pushupRepMinAngle!, elbowAngle);
      }

      // 达到过目标深度即可，不要求在最低点连续停留。
      if (_pushupDown &&
          elbowAngle <= (t.pushupDownAngle + t.pushupDepthMargin * 0.55)) {
        _pushupRepReachedDepth = true;
      }

      final depthIssue = _stableIssue(
        type: PoseErrorType.pushupDepthNotEnough,
        active: _pushupDown &&
            !_pushupRepReachedDepth &&
            _pushupRepMinAngle != null &&
            elbowAngle > _pushupRepMinAngle! + 8 &&
            elbowAngle > t.pushupDownAngle + t.pushupDepthMargin * 0.75,
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
      // 与深蹲相同，短暂丢帧先保留状态，持续丢失再重置。
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

  /// 分析平板支撑动作，在检查身体稳定性的同时累计有效时长。
  ExerciseAnalysisResult _analyzePlank(
      Pose pose,
      PoseMetricSnapshot snapshot,
      PersonalizedThresholds t,
      UserProfile profile,
      ) {
    final issues = <PoseIssue>[];

    if (!_isFloorBodyPose(pose)) {
      _lastPlankTimestamp = null;
      return ExerciseAnalysisResult(
        feedback: '请先摆出平板支撑姿势',
        score: 0,
        count: _plankHoldSeconds.floor(),
        countDelta: 0,
        repJustCounted: false,
        repJustCountedClean: false,
        issues: const <PoseIssue>[],
        depthModeLabel: '单目 3D',
        metrics: <String, double>{
          'hold_seconds': _plankHoldSeconds,
        },
      );
    }

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

    // 平板支撑没有“次数”，这里只判断身体是否保持在中立区间。
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

    // 只有当前没有明显问题时，才累计本段支撑时长。
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

  /// 重置指定动作的计数器和过程状态，开始一轮新的识别。
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
    // 每个问题按严重程度扣分，并保留一个最低分，避免体验过于极端。
    final deduction = issues.fold<double>(
      0,
      (sum, item) => sum + item.severity * 22,
    );
    return (100 - deduction).clamp(35, 100).round();
  }

  List<PoseIssue> _sortedIssues(List<PoseIssue> issues) {
    // 让最严重的问题排在最前面，方便 UI 和语音优先提醒。
    final sorted = <PoseIssue>[...issues]
      ..sort((a, b) => b.severity.compareTo(a.severity));
    return sorted;
  }

  String _feedbackFromIssues(List<PoseIssue> issues, {required String fallback}) {
    if (issues.isEmpty) {
      return fallback;
    }
    // 当前只返回最严重问题的文案，避免同时提示太多内容。
    final sorted = <PoseIssue>[...issues]
      ..sort((a, b) => b.severity.compareTo(a.severity));
    return sorted.first.message;
  }

  /// 用连续帧锁存问题状态，避免提示在抖动数据中频繁闪烁。
  PoseIssue? _stableIssue({
    required PoseErrorType type,
    required bool active,
    required PoseIssue Function() build,
    int onFrames = _issueOnFrames,
    int offFrames = _issueOffFrames,
  }) {
    if (active) {
      // 连续多帧都为 active，才真正挂上这个问题。
      _issueOnStreak[type] = (_issueOnStreak[type] ?? 0) + 1;
      _issueOffStreak[type] = 0;
      if ((_issueOnStreak[type] ?? 0) >= onFrames) {
        _latchedIssues.add(type);
      }
    } else {
      // 反过来，连续若干帧恢复正常后，再把问题撤掉。
      _issueOnStreak[type] = 0;
      _issueOffStreak[type] = (_issueOffStreak[type] ?? 0) + 1;
      if ((_issueOffStreak[type] ?? 0) >= offFrames) {
        _latchedIssues.remove(type);
      }
    }
    return _latchedIssues.contains(type) ? build() : null;
  }

  /// 对指标做 EMA 平滑，减少抖动对计数和反馈的影响。
  double? _smoothMetric(String key, double? value, {double alpha = _emaFast}) {
    if (value == null || value.isNaN || value.isInfinite) {
      return null;
    }
    final previous = _emaMetrics[key];
    // EMA: 新值保留一部分，旧值保留一部分，减少抖动。
    final smoothed =
        previous == null ? value : previous * (1 - alpha) + value * alpha;
    _emaMetrics[key] = smoothed;
    return smoothed;
  }

  /// 判断深蹲是否已经明显起身，只有满足条件才允许计数。
  bool _hasSquatRecoveredEnough(
    double kneeAngle,
    PersonalizedThresholds t,
    bool isMoveNet,
  ) {
    final minAngle = _squatRepMinAngle;
    // 两种通过方式：
    // 1. 已经接近站直；
    // 2. 虽未完全站直，但相比最低点已经明显回弹。
    final nearTop = kneeAngle >= (t.squatUpAngle - (isMoveNet ? 24 : 16));
    final clearRebound = minAngle != null &&
        kneeAngle >=
            math.max(
              minAngle + (isMoveNet ? 10 : 18),
              t.squatDownAngle + (isMoveNet ? 8 : 14),
            );
    return nearTop || clearRebound;
  }

  /// 判断俯卧撑是否已经明显推起，只有满足条件才允许计数。
  bool _hasPushupRecoveredEnough(
    double elbowAngle,
    PersonalizedThresholds t,
    bool isMoveNet,
  ) {
    final minAngle = _pushupRepMinAngle;
    // 两种通过方式：
    // 1. 手肘基本重新伸直；
    // 2. 相比最低点已经明显推起。
    final nearTop = elbowAngle >= (t.pushupUpAngle - (isMoveNet ? 18 : 16));
    final clearRebound = minAngle != null &&
        elbowAngle >=
            math.max(
              minAngle + (isMoveNet ? 18 : 22),
              t.pushupDownAngle + (isMoveNet ? 14 : 16),
            );
    return nearTop || clearRebound;
  }

  bool _isFloorBodyPose(Pose pose) {
    final isMoveNet = pose.source.startsWith('movenet');
    final threshold = isMoveNet ? 0.34 : 0.45;

    bool reliable(PoseLandmark? p) {
      return p != null && p.likelihood >= threshold;
    }

    final leftShoulder = pose[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose[PoseLandmarkType.rightShoulder];
    final leftHip = pose[PoseLandmarkType.leftHip];
    final rightHip = pose[PoseLandmarkType.rightHip];
    final leftAnkle = pose[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose[PoseLandmarkType.rightAnkle];

    if (!reliable(leftShoulder) ||
        !reliable(rightShoulder) ||
        !reliable(leftHip) ||
        !reliable(rightHip)) {
      return false;
    }

    final shoulderX = (leftShoulder!.x + rightShoulder!.x) / 2;
    final shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    final hipX = (leftHip!.x + rightHip!.x) / 2;
    final hipY = (leftHip.y + rightHip.y) / 2;

    final torsoDx = (shoulderX - hipX).abs();
    final torsoDy = (shoulderY - hipY).abs();
    final torsoLen = math.sqrt(torsoDx * torsoDx + torsoDy * torsoDy);

    if (torsoLen < 25) return false;

    final torsoLooksHorizontal = torsoDx > torsoDy * 0.75;

    if (!reliable(leftAnkle) && !reliable(rightAnkle)) {
      return torsoLooksHorizontal;
    }

    final anklePoints = <PoseLandmark>[
      if (reliable(leftAnkle)) leftAnkle!,
      if (reliable(rightAnkle)) rightAnkle!,
    ];

    final ankleX =
        anklePoints.map((p) => p.x).reduce((a, b) => a + b) / anklePoints.length;
    final ankleY =
        anklePoints.map((p) => p.y).reduce((a, b) => a + b) / anklePoints.length;

    final bodyDx = (shoulderX - ankleX).abs();
    final bodyDy = (shoulderY - ankleY).abs();

    final bodyLooksHorizontal = bodyDx > bodyDy * 0.65;

    return torsoLooksHorizontal && bodyLooksHorizontal;
  }

  /// 规范化外部传入的视角标签，只保留系统支持的取值。
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

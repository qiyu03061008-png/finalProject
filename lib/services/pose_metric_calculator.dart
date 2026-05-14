import 'dart:math';

import '../models/pose_landmark.dart';

class PoseMetricSnapshot {
  const PoseMetricSnapshot({
    required this.metrics,
    required this.inferredViewTag,
  });

  // 已计算好的指标集合，供分析器按 key 直接读取。
  final Map<String, double> metrics;
  final String inferredViewTag;

  double? operator [](String key) => metrics[key];
}

class PoseMetricCalculator {
  const PoseMetricCalculator();

  static const double minLikelihood = 0.45;
  static const double minLikelihoodForRep = 0.30;

  /// 从单帧姿态中提取后续动作分析所依赖的各项指标。
  PoseMetricSnapshot calculate(Pose pose) {
    // 先建立身体局部坐标系，后面很多 3D 指标都依赖它。
    final frame = _buildBodyFrame(pose);
    final leftKneeAngle = _jointAngleForRep(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.leftKnee],
      pose[PoseLandmarkType.leftAnkle],
    );
    final rightKneeAngle = _jointAngleForRep(
      pose[PoseLandmarkType.rightHip],
      pose[PoseLandmarkType.rightKnee],
      pose[PoseLandmarkType.rightAnkle],
    );
    final kneeAngleMetric = _combineSideMetrics(
      _metricFromChain(
        leftKneeAngle,
        <PoseLandmark?>[
          pose[PoseLandmarkType.leftHip],
          pose[PoseLandmarkType.leftKnee],
          pose[PoseLandmarkType.leftAnkle],
        ],
      ),
      _metricFromChain(
        rightKneeAngle,
        <PoseLandmark?>[
          pose[PoseLandmarkType.rightHip],
          pose[PoseLandmarkType.rightKnee],
          pose[PoseLandmarkType.rightAnkle],
        ],
      ),
      disagreementToleranceDeg: 18,
      singleSidePenalty: 0.88,
    );

    final leftBodyLineAngle = _jointAngleForRep(
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.leftAnkle],
    );
    final rightBodyLineAngle = _jointAngleForRep(
      pose[PoseLandmarkType.rightShoulder],
      pose[PoseLandmarkType.rightHip],
      pose[PoseLandmarkType.rightAnkle],
    );
    final bodyLineMetric = _combineSideMetrics(
      _metricFromChain(
        leftBodyLineAngle,
        <PoseLandmark?>[
          pose[PoseLandmarkType.leftShoulder],
          pose[PoseLandmarkType.leftHip],
          pose[PoseLandmarkType.leftAnkle],
        ],
      ),
      _metricFromChain(
        rightBodyLineAngle,
        <PoseLandmark?>[
          pose[PoseLandmarkType.rightShoulder],
          pose[PoseLandmarkType.rightHip],
          pose[PoseLandmarkType.rightAnkle],
        ],
      ),
      disagreementToleranceDeg: 14,
      singleSidePenalty: 0.85,
    );

    final torsoLeanMetric = _torsoForwardLeanDeg(pose, frame);
    final kneeValgusMetric =
        frame == null ? null : _kneeValgusAngle3D(pose, frame);
    final shankLeanMetric =
        frame == null ? null : _shankForwardLeanDeg3D(pose, frame);
    final pushupElbowMetric = _combineSideMetrics(
      _metricFromChain(
        _jointAngleForRep(
          pose[PoseLandmarkType.leftShoulder],
          pose[PoseLandmarkType.leftElbow],
          pose[PoseLandmarkType.leftWrist],
        ),
        <PoseLandmark?>[
          pose[PoseLandmarkType.leftShoulder],
          pose[PoseLandmarkType.leftElbow],
          pose[PoseLandmarkType.leftWrist],
        ],
      ),
      _metricFromChain(
        _jointAngleForRep(
          pose[PoseLandmarkType.rightShoulder],
          pose[PoseLandmarkType.rightElbow],
          pose[PoseLandmarkType.rightWrist],
        ),
        <PoseLandmark?>[
          pose[PoseLandmarkType.rightShoulder],
          pose[PoseLandmarkType.rightElbow],
          pose[PoseLandmarkType.rightWrist],
        ],
      ),
      disagreementToleranceDeg: 20,
      singleSidePenalty: 0.9,
    );
    final hipOffsetMetric =
        frame == null ? null : _bodyLineHipOffset(pose, frame);
    final elbowFlareMetric =
        frame == null ? null : _elbowFlareMetric3D(pose, frame);
    final neckMetric = frame == null ? null : _neckNeutralAngle3D(pose, frame);

    final metrics = <String, double>{};
    void addMetric(String key, double? value) {
      // 统一过滤掉无效数值，避免下游再做重复判空。
      if (value != null && value.isFinite && !value.isNaN) {
        metrics[key] = value;
      }
    }

    addMetric('squatKneeAngle', kneeAngleMetric.value);
    addMetric('squatKneeAngleConfidence', kneeAngleMetric.confidence);
    addMetric('squatTorsoLeanDeg', torsoLeanMetric?.value);
    addMetric('squatTorsoLeanConfidence', torsoLeanMetric?.confidence);
    addMetric('squatKneeValgusAngle', kneeValgusMetric?.value);
    addMetric('squatKneeValgusConfidence', kneeValgusMetric?.confidence);
    addMetric(
      'squatKneeValgusDisagreementDeg',
      kneeValgusMetric?.disagreementDeg,
    );
    addMetric('squatShankLeanDeg', shankLeanMetric?.value);
    addMetric('squatShankLeanConfidence', shankLeanMetric?.confidence);
    addMetric(
      'squatShankLeanDisagreementDeg',
      shankLeanMetric?.disagreementDeg,
    );
    addMetric('pushupElbowAngle', pushupElbowMetric.value);
    addMetric('pushupElbowAngleConfidence', pushupElbowMetric.confidence);
    addMetric('pushupBodyLineAngle', bodyLineMetric.value);
    addMetric('pushupBodyLineConfidence', bodyLineMetric.confidence);
    addMetric(
      'pushupBodyLineDeviation',
      _deviationFromStraight(bodyLineMetric.value),
    );
    addMetric('pushupHipOffset', hipOffsetMetric?.value);
    addMetric('pushupHipOffsetConfidence', hipOffsetMetric?.confidence);
    addMetric('pushupElbowFlareDeg', elbowFlareMetric?.value);
    addMetric('pushupElbowFlareConfidence', elbowFlareMetric?.confidence);
    addMetric(
      'pushupElbowFlareDisagreementDeg',
      elbowFlareMetric?.disagreementDeg,
    );
    addMetric('plankBodyLineAngle', bodyLineMetric.value);
    addMetric('plankBodyLineConfidence', bodyLineMetric.confidence);
    addMetric(
      'plankBodyLineDeviation',
      _deviationFromStraight(bodyLineMetric.value),
    );
    addMetric('plankHipOffset', hipOffsetMetric?.value);
    addMetric('plankHipOffsetConfidence', hipOffsetMetric?.confidence);
    addMetric('plankNeckAngle', neckMetric?.value);
    addMetric('plankNeckConfidence', neckMetric?.confidence);

    return PoseMetricSnapshot(
      metrics: metrics,
      inferredViewTag: inferViewTag(pose),
    );
  }

  /// 估计当前姿态更接近正面、侧面还是斜侧面视角。
  String inferViewTag(Pose pose) {
    final leftShoulder = pose[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose[PoseLandmarkType.rightShoulder];
    final leftHip = pose[PoseLandmarkType.leftHip];
    final rightHip = pose[PoseLandmarkType.rightHip];

    if (!_isReliable(leftShoulder) ||
        !_isReliable(rightShoulder) ||
        !_isReliable(leftHip) ||
        !_isReliable(rightHip)) {
      return 'front';
    }

    final shoulderCenter = _midpoint(leftShoulder, rightShoulder);
    final hipCenter = _midpoint(leftHip, rightHip);
    if (shoulderCenter == null || hipCenter == null) {
      return 'front';
    }

    final shoulderWidth = _distance2(leftShoulder!, rightShoulder!);
    final hipWidth = _distance2(leftHip!, rightHip!);
    final torsoLength = _distance2(shoulderCenter, hipCenter);
    if (torsoLength < 1e-5) {
      return 'front';
    }

    // 躯干越“窄”，越像侧面；越“宽”，越像正面。
    final widthRatio = ((shoulderWidth + hipWidth) / 2.0) / torsoLength;
    if (widthRatio <= 0.32) {
      return 'side';
    }
    if (widthRatio <= 0.6) {
      return 'oblique';
    }
    return 'front';
  }

  /// 计算关节夹角，优先使用 3D 点位，必要时退回到 2D 计算。
  double? _jointAngleForRep(
    PoseLandmark? p1,
    PoseLandmark? p2,
    PoseLandmark? p3,
  ) {
    if (!_isReliableRep(p1) || !_isReliableRep(p2) || !_isReliableRep(p3)) {
      return null;
    }
    final a = p1!;
    final b = p2!;
    final c = p3!;

    // 以中间点 p2 为关节顶点，分别连向两侧点计算夹角。
    final v1 = _vector3(b, a);
    final v2 = _vector3(b, c);
    // 优先使用 3D 角度；深度可用时，它比纯 2D 更稳。
    final angle3d = _vectorAngle3(v1, v2);
    if (!angle3d.isNaN && angle3d.isFinite) {
      return angle3d;
    }

    // 如果 3D 不稳定，则退回到图像平面的 2D 夹角。
    final v1x = a.x - b.x;
    final v1y = a.y - b.y;
    final v2x = c.x - b.x;
    final v2y = c.y - b.y;
    final m1 = sqrt(v1x * v1x + v1y * v1y);
    final m2 = sqrt(v2x * v2x + v2y * v2y);
    if (m1 < 1e-5 || m2 < 1e-5) {
      return null;
    }
    final cosValue = (v1x * v2x + v1y * v2y) / (m1 * m2);
    return acos(cosValue.clamp(-1.0, 1.0)) * 180 / pi;
  }

  /// 构建以人体为中心的局部坐标系，供 3D 姿态指标使用。
  _BodyFrame? _buildBodyFrame(Pose pose) {
    final leftHip = pose[PoseLandmarkType.leftHip];
    final rightHip = pose[PoseLandmarkType.rightHip];
    final leftShoulder = pose[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose[PoseLandmarkType.rightShoulder];
    if (!_isReliable(leftHip) ||
        !_isReliable(rightHip) ||
        !_isReliable(leftShoulder) ||
        !_isReliable(rightShoulder)) {
      return null;
    }

    final hipCenter = _midpoint(leftHip, rightHip);
    final shoulderCenter = _midpoint(leftShoulder, rightShoulder);
    if (hipCenter == null || shoulderCenter == null) {
      return null;
    }

    // lateral: 身体左右方向；up: 身体向上方向；forward: 身体朝前方向。
    var lateral = _vector3(leftHip!, rightHip!);
    if (lateral.magnitude < 1e-5) {
      lateral = _vector3(leftShoulder!, rightShoulder!);
    }
    var up = _vector3(hipCenter, shoulderCenter);
    if (up.magnitude < 1e-5) {
      return null;
    }

    lateral = lateral.normalized;
    up = up.normalized;

    // 通过叉乘补出第三个正交方向，形成身体自己的坐标系。
    var forward = lateral.cross(up);
    if (forward.magnitude < 1e-5) {
      return null;
    }
    forward = forward.normalized;
    up = forward.cross(lateral).normalized;

    return _BodyFrame(
      origin: hipCenter,
      lateral: lateral,
      up: up,
      forward: forward,
    );
  }

  /// 计算躯干向量与下肢支撑向量在矢状面的3D夹角。
  _CombinedMetric? _torsoForwardLeanDeg(Pose pose, _BodyFrame? frame) {
    if (frame == null) {
      return null;
    }
    final shoulderCenter = _midpoint(
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.rightShoulder],
    );
    final hipCenter = _midpoint(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    );
    final ankleCenter = _midpoint(
      pose[PoseLandmarkType.leftAnkle],
      pose[PoseLandmarkType.rightAnkle],
    );
    if (!_isReliable(shoulderCenter) ||
        !_isReliable(hipCenter) ||
        !_isReliable(ankleCenter)) {
      return null;
    }

    final torso =
        _projectToSagittal(_vector3(hipCenter!, shoulderCenter!), frame);
    final support =
        _projectToSagittal(_vector3(ankleCenter!, hipCenter), frame);
    if (torso.magnitude < 1e-5 || support.magnitude < 1e-5) {
      return null;
    }

    final angle = _vectorAngle3(support, torso);
    if (angle.isNaN || !angle.isFinite) {
      return null;
    }
    return _CombinedMetric(
      value: angle,
      confidence: _chainConfidence(<PoseLandmark?>[
        pose[PoseLandmarkType.leftShoulder],
        pose[PoseLandmarkType.rightShoulder],
        pose[PoseLandmarkType.leftHip],
        pose[PoseLandmarkType.rightHip],
        pose[PoseLandmarkType.leftAnkle],
        pose[PoseLandmarkType.rightAnkle],
      ]),
    );
  }

  _CombinedMetric? _kneeValgusAngle3D(Pose pose, _BodyFrame frame) {
    final leftHip = pose[PoseLandmarkType.leftHip];
    final rightHip = pose[PoseLandmarkType.rightHip];
    final leftKnee = pose[PoseLandmarkType.leftKnee];
    final rightKnee = pose[PoseLandmarkType.rightKnee];
    final leftAnkle = pose[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose[PoseLandmarkType.rightAnkle];
    if (!_isReliable(leftHip) ||
        !_isReliable(rightHip) ||
        !_isReliable(leftKnee) ||
        !_isReliable(rightKnee) ||
        !_isReliable(leftAnkle) ||
        !_isReliable(rightAnkle)) {
      return null;
    }

    return _combineSideMetrics(
      _metricFromChain(
        _frontalAlignmentAngle(leftHip!, leftKnee!, leftAnkle!, frame),
        <PoseLandmark?>[leftHip, leftKnee, leftAnkle],
      ),
      _metricFromChain(
        _frontalAlignmentAngle(rightHip!, rightKnee!, rightAnkle!, frame),
        <PoseLandmark?>[rightHip, rightKnee, rightAnkle],
      ),
      disagreementToleranceDeg: 16,
      singleSidePenalty: 0.8,
    );
  }

  _CombinedMetric? _shankForwardLeanDeg3D(Pose pose, _BodyFrame frame) {
    final leftKnee = pose[PoseLandmarkType.leftKnee];
    final rightKnee = pose[PoseLandmarkType.rightKnee];
    final leftAnkle = pose[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose[PoseLandmarkType.rightAnkle];
    if (!_isReliable(leftKnee) ||
        !_isReliable(rightKnee) ||
        !_isReliable(leftAnkle) ||
        !_isReliable(rightAnkle)) {
      return null;
    }

    return _combineSideMetrics(
      _metricFromChain(
        _segmentLeanFromUp(
          _projectToSagittal(_vector3(leftAnkle!, leftKnee!), frame),
          frame,
        ),
        <PoseLandmark?>[leftKnee, leftAnkle],
      ),
      _metricFromChain(
        _segmentLeanFromUp(
          _projectToSagittal(_vector3(rightAnkle!, rightKnee!), frame),
          frame,
        ),
        <PoseLandmark?>[rightKnee, rightAnkle],
      ),
      disagreementToleranceDeg: 12,
      singleSidePenalty: 0.82,
    );
  }

  /// 计算髋部偏离肩到踝参考线的程度。
  _CombinedMetric? _bodyLineHipOffset(Pose pose, _BodyFrame frame) {
    final shoulderCenter = _midpoint(
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.rightShoulder],
    );
    final hipCenter = _midpoint(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    );
    final ankleCenter = _midpoint(
      pose[PoseLandmarkType.leftAnkle],
      pose[PoseLandmarkType.rightAnkle],
    );
    if (!_isReliable(shoulderCenter) ||
        !_isReliable(hipCenter) ||
        !_isReliable(ankleCenter)) {
      return null;
    }

    final s = _sagittalPoint(shoulderCenter!, frame);
    final h = _sagittalPoint(hipCenter!, frame);
    final a = _sagittalPoint(ankleCenter!, frame);
    final df = a.forward - s.forward;
    final lineLen = sqrt(df * df + pow(a.up - s.up, 2));
    if (lineLen < 1e-5) {
      return null;
    }

    // 把髋部投影到“肩到踝”的参考线上，再看它偏离了多少。
    final t =
        df.abs() < 1e-5 ? 0.5 : ((h.forward - s.forward) / df).clamp(0.0, 1.0);
    final expectedUp = s.up + (a.up - s.up) * t;
    return _CombinedMetric(
      value: (h.up - expectedUp) / lineLen,
      confidence: _chainConfidence(<PoseLandmark?>[
        pose[PoseLandmarkType.leftShoulder],
        pose[PoseLandmarkType.rightShoulder],
        pose[PoseLandmarkType.leftHip],
        pose[PoseLandmarkType.rightHip],
        pose[PoseLandmarkType.leftAnkle],
        pose[PoseLandmarkType.rightAnkle],
      ]),
    );
  }

  _CombinedMetric? _elbowFlareMetric3D(Pose pose, _BodyFrame frame) {
    final shoulderCenter = _midpoint(
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.rightShoulder],
    );
    final hipCenter = _midpoint(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    );
    final leftElbow = pose[PoseLandmarkType.leftElbow];
    final rightElbow = pose[PoseLandmarkType.rightElbow];
    final leftShoulder = pose[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose[PoseLandmarkType.rightShoulder];
    if (!_isReliable(shoulderCenter) ||
        !_isReliable(hipCenter) ||
        !_isReliable(leftElbow) ||
        !_isReliable(rightElbow) ||
        !_isReliable(leftShoulder) ||
        !_isReliable(rightShoulder)) {
      return null;
    }

    // 在额状面里比较“躯干方向”和“上臂方向”的夹角，用来判断手肘外展。
    final torso =
        _projectToFrontal(_vector3(hipCenter!, shoulderCenter!), frame);
    return _combineSideMetrics(
      _metricFromChain(
        _vectorAngle3(
          torso,
          _projectToFrontal(_vector3(leftShoulder!, leftElbow!), frame),
        ),
        <PoseLandmark?>[leftShoulder, leftElbow],
      ),
      _metricFromChain(
        _vectorAngle3(
          torso,
          _projectToFrontal(_vector3(rightShoulder!, rightElbow!), frame),
        ),
        <PoseLandmark?>[rightShoulder, rightElbow],
      ),
      disagreementToleranceDeg: 18,
      singleSidePenalty: 0.8,
    );
  }

  _CombinedMetric? _neckNeutralAngle3D(Pose pose, _BodyFrame frame) {
    final earCenter = _midpoint(
      pose[PoseLandmarkType.leftEar],
      pose[PoseLandmarkType.rightEar],
    );
    final shoulderCenter = _midpoint(
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.rightShoulder],
    );
    final hipCenter = _midpoint(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    );
    if (!_isReliable(earCenter) ||
        !_isReliable(shoulderCenter) ||
        !_isReliable(hipCenter)) {
      return null;
    }

    // 比较“肩到耳”和“肩到髋”的夹角，近似描述颈部是否中立。
    final neck =
        _projectToSagittal(_vector3(shoulderCenter!, earCenter!), frame);
    final torso =
        _projectToSagittal(_vector3(shoulderCenter, hipCenter!), frame);
    final angle = _vectorAngle3(neck, torso);
    if (angle.isNaN || !angle.isFinite) {
      return null;
    }
    return _CombinedMetric(
      value: angle,
      confidence: _chainConfidence(<PoseLandmark?>[
        pose[PoseLandmarkType.leftEar],
        pose[PoseLandmarkType.rightEar],
        pose[PoseLandmarkType.leftShoulder],
        pose[PoseLandmarkType.rightShoulder],
        pose[PoseLandmarkType.leftHip],
        pose[PoseLandmarkType.rightHip],
      ]),
    );
  }

  double? _deviationFromStraight(double? angleDeg) {
    if (angleDeg == null) {
      return null;
    }
    // 180 度表示近似一条直线，偏差越大说明越弯。
    return (180 - angleDeg).abs();
  }

  PoseLandmark? _midpoint(PoseLandmark? a, PoseLandmark? b) {
    if (!_isReliable(a) || !_isReliable(b)) {
      return null;
    }
    // 中点的置信度取两者较小值，避免过度乐观。
    final pa = a!;
    final pb = b!;
    return PoseLandmark(
      x: (pa.x + pb.x) / 2,
      y: (pa.y + pb.y) / 2,
      z: (pa.z + pb.z) / 2,
      likelihood: min(pa.likelihood, pb.likelihood),
    );
  }

  double _distance2(PoseLandmark a, PoseLandmark b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
  }

  double? _avg(double? a, double? b) {
    // 左右身体两侧能算几个就算几个，都有时再取平均。
    if (a == null && b == null) {
      return null;
    }
    if (a == null) {
      return b;
    }
    if (b == null) {
      return a;
    }
    return (a + b) / 2.0;
  }

  double _lateralCoord(
      PoseLandmark point, _BodyFrame frame, PoseLandmark origin) {
    return _vector3(origin, point).dot(frame.lateral);
  }

  double _forwardCoord(PoseLandmark point, _BodyFrame frame) {
    return _vector3(frame.origin, point).dot(frame.forward);
  }

  _PlanePoint _sagittalPoint(PoseLandmark point, _BodyFrame frame) {
    final v = _vector3(frame.origin, point);
    return _PlanePoint(v.dot(frame.forward), v.dot(frame.up));
  }

  _V3 _vector3(PoseLandmark from, PoseLandmark to) {
    return _V3(to.x - from.x, to.y - from.y, to.z - from.z);
  }

  _V3 _projectToSagittal(_V3 v, _BodyFrame frame) {
    return frame.forward * v.dot(frame.forward) + frame.up * v.dot(frame.up);
  }

  _V3 _projectToFrontal(_V3 v, _BodyFrame frame) {
    return frame.lateral * v.dot(frame.lateral) + frame.up * v.dot(frame.up);
  }

  double _vectorAngle3(_V3 v1, _V3 v2) {
    final m1 = v1.magnitude;
    final m2 = v2.magnitude;
    if (m1 < 1e-5 || m2 < 1e-5) {
      return double.nan;
    }
    final cosValue = (v1.dot(v2) / (m1 * m2)).clamp(-1.0, 1.0);
    return acos(cosValue) * 180 / pi;
  }

  double _signedAngleOnPlane(_V3 base, _V3 target, _V3 normal) {
    final baseN = base.normalized;
    final targetN = target.normalized;
    final cross = baseN.cross(targetN);
    final dot = baseN.dot(targetN).clamp(-1.0, 1.0);
    final sign = cross.dot(normal) >= 0 ? 1.0 : -1.0;
    return atan2(cross.magnitude * sign, dot) * 180 / pi;
  }

  double? _segmentLeanFromUp(_V3 segment, _BodyFrame frame) {
    final angle = _vectorAngle3(segment, frame.up);
    if (angle.isNaN || !angle.isFinite) {
      return null;
    }
    return angle;
  }

  double? _frontalAlignmentAngle(
    PoseLandmark hip,
    PoseLandmark knee,
    PoseLandmark ankle,
    _BodyFrame frame,
  ) {
    final thigh = _projectToFrontal(_vector3(knee, hip), frame);
    final shank = _projectToFrontal(_vector3(knee, ankle), frame);
    final angle = _vectorAngle3(thigh, shank);
    if (angle.isNaN || !angle.isFinite) {
      return null;
    }
    return angle;
  }

  _SideMetric? _metricFromChain(
    double? value,
    List<PoseLandmark?> points,
  ) {
    if (value == null || value.isNaN || !value.isFinite) {
      return null;
    }
    final confidence = _chainConfidence(points);
    if (confidence <= 0) {
      return null;
    }
    return _SideMetric(value: value, confidence: confidence);
  }

  _CombinedMetric _combineSideMetrics(
    _SideMetric? left,
    _SideMetric? right, {
    required double disagreementToleranceDeg,
    double singleSidePenalty = 0.85,
  }) {
    if (left == null && right == null) {
      return const _CombinedMetric();
    }
    if (left == null) {
      return _CombinedMetric(
        value: right!.value,
        confidence: right.confidence * singleSidePenalty,
      );
    }
    if (right == null) {
      return _CombinedMetric(
        value: left.value,
        confidence: left.confidence * singleSidePenalty,
      );
    }

    final disagreement = (left.value - right.value).abs();
    if (disagreement > disagreementToleranceDeg * 1.6) {
      final preferred = left.confidence >= right.confidence ? left : right;
      return _CombinedMetric(
        value: preferred.value,
        confidence: preferred.confidence * 0.18,
        disagreementDeg: disagreement,
      );
    }

    if (disagreement > disagreementToleranceDeg) {
      final preferred = left.confidence >= right.confidence ? left : right;
      return _CombinedMetric(
        value: preferred.value,
        confidence: preferred.confidence * 0.55,
        disagreementDeg: disagreement,
      );
    }

    final weightSum = left.confidence + right.confidence;
    final value = weightSum <= 1e-6
        ? (left.value + right.value) / 2.0
        : (left.value * left.confidence + right.value * right.confidence) /
            weightSum;
    final agreementFactor =
        (1 - disagreement / (disagreementToleranceDeg * 1.6)).clamp(0.45, 1.0);
    return _CombinedMetric(
      value: value,
      confidence:
          (((left.confidence + right.confidence) / 2.0) * agreementFactor)
              .clamp(0.0, 1.0),
      disagreementDeg: disagreement,
    );
  }

  double _chainConfidence(List<PoseLandmark?> points) {
    var minLike = 1.0;
    var hasPoint = false;
    for (final point in points) {
      if (point == null || point.likelihood < minLikelihood) {
        return 0;
      }
      hasPoint = true;
      minLike = min(minLike, point.likelihood);
    }
    if (!hasPoint) {
      return 0;
    }
    return ((minLike - minLikelihood) / (1 - minLikelihood)).clamp(0.0, 1.0);
  }

  bool _isReliable(PoseLandmark? point) {
    return point != null && point.likelihood >= minLikelihood;
  }

  bool _isReliableRep(PoseLandmark? point) {
    return point != null && point.likelihood >= minLikelihoodForRep;
  }
}

class _BodyFrame {
  const _BodyFrame({
    required this.origin,
    required this.lateral,
    required this.up,
    required this.forward,
  });

  final PoseLandmark origin;
  final _V3 lateral;
  final _V3 up;
  final _V3 forward;
}

class _PlanePoint {
  const _PlanePoint(this.forward, this.up);

  final double forward;
  final double up;
}

class _SideMetric {
  const _SideMetric({
    required this.value,
    required this.confidence,
  });

  final double value;
  final double confidence;
}

class _CombinedMetric {
  const _CombinedMetric({
    this.value,
    this.confidence = 0,
    this.disagreementDeg,
  });

  final double? value;
  final double confidence;
  final double? disagreementDeg;
}

class _V3 {
  const _V3(this.x, this.y, this.z);

  final double x;
  final double y;
  final double z;

  double get magnitude => sqrt(x * x + y * y + z * z);

  _V3 get normalized {
    final m = magnitude;
    if (m < 1e-8) {
      return const _V3(0, 0, 0);
    }
    return _V3(x / m, y / m, z / m);
  }

  double dot(_V3 other) => x * other.x + y * other.y + z * other.z;

  _V3 cross(_V3 other) => _V3(
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x,
      );

  _V3 operator +(_V3 other) => _V3(x + other.x, y + other.y, z + other.z);

  _V3 operator -(_V3 other) => _V3(x - other.x, y - other.y, z - other.z);

  _V3 operator *(double scale) => _V3(x * scale, y * scale, z * scale);
}

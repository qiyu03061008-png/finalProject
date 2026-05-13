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
    final kneeAngle = _avg(
      _jointAngleForRep(
        pose[PoseLandmarkType.leftHip],
        pose[PoseLandmarkType.leftKnee],
        pose[PoseLandmarkType.leftAnkle],
      ),
      _jointAngleForRep(
        pose[PoseLandmarkType.rightHip],
        pose[PoseLandmarkType.rightKnee],
        pose[PoseLandmarkType.rightAnkle],
      ),
    );

    final bodyLineAngle = _avg(
      _jointAngleForRep(
        pose[PoseLandmarkType.leftShoulder],
        pose[PoseLandmarkType.leftHip],
        pose[PoseLandmarkType.leftAnkle],
      ),
      _jointAngleForRep(
        pose[PoseLandmarkType.rightShoulder],
        pose[PoseLandmarkType.rightHip],
        pose[PoseLandmarkType.rightAnkle],
      ),
    );

    final metrics = <String, double>{};
    void addMetric(String key, double? value) {
      // 统一过滤掉无效数值，避免下游再做重复判空。
      if (value != null && value.isFinite && !value.isNaN) {
        metrics[key] = value;
      }
    }

    addMetric('squatKneeAngle', kneeAngle);
    addMetric('squatTorsoLeanDeg', _torsoForwardLeanDeg(pose, frame));
    addMetric(
      'squatKneeInwardRatio',
      frame == null ? null : _kneeInwardRatio3D(pose, frame),
    );
    addMetric(
      'squatKneeOverToe',
      frame == null ? null : _kneeOverToeMetric3D(pose, frame),
    );
    addMetric(
      'pushupElbowAngle',
      _avg(
        _jointAngleForRep(
          pose[PoseLandmarkType.leftShoulder],
          pose[PoseLandmarkType.leftElbow],
          pose[PoseLandmarkType.leftWrist],
        ),
        _jointAngleForRep(
          pose[PoseLandmarkType.rightShoulder],
          pose[PoseLandmarkType.rightElbow],
          pose[PoseLandmarkType.rightWrist],
        ),
      ),
    );
    addMetric('pushupBodyLineAngle', bodyLineAngle);
    addMetric('pushupBodyLineDeviation', _deviationFromStraight(bodyLineAngle));
    addMetric(
      'pushupHipOffset',
      frame == null ? null : _bodyLineHipOffset(pose, frame),
    );
    addMetric(
      'pushupElbowFlareDeg',
      frame == null ? null : _elbowFlareMetric3D(pose, frame),
    );
    addMetric('plankBodyLineAngle', bodyLineAngle);
    addMetric('plankBodyLineDeviation', _deviationFromStraight(bodyLineAngle));
    addMetric(
      'plankHipOffset',
      frame == null ? null : _bodyLineHipOffset(pose, frame),
    );
    addMetric(
      'plankNeckAngle',
      frame == null ? null : _neckNeutralAngleStable(pose, frame),
    );

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

  /// 计算躯干相对下肢支撑线的前倾角度。
  double? _torsoForwardLeanDeg(Pose pose, _BodyFrame? frame) {
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

    final signed = _signedAngleOnPlane(support, torso, frame.lateral);
    return max(0.0, signed);
  }

  double? _kneeInwardRatio3D(Pose pose, _BodyFrame frame) {
    final hipCenter = _midpoint(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    );
    final leftKnee = pose[PoseLandmarkType.leftKnee];
    final rightKnee = pose[PoseLandmarkType.rightKnee];
    final leftAnkle = pose[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose[PoseLandmarkType.rightAnkle];
    if (!_isReliable(hipCenter) ||
        !_isReliable(leftKnee) ||
        !_isReliable(rightKnee) ||
        !_isReliable(leftAnkle) ||
        !_isReliable(rightAnkle)) {
      return null;
    }

    final leftKneeLat = _lateralCoord(leftKnee!, frame, hipCenter!);
    final rightKneeLat = _lateralCoord(rightKnee!, frame, hipCenter);
    final leftAnkleLat = _lateralCoord(leftAnkle!, frame, hipCenter);
    final rightAnkleLat = _lateralCoord(rightAnkle!, frame, hipCenter);
    if (leftAnkleLat.abs() < 1e-5 || rightAnkleLat.abs() < 1e-5) {
      return null;
    }

    // 膝盖横向位置与脚踝横向位置越接近，通常说明膝盖没有明显内扣。
    final leftRatio = leftKneeLat.abs() / leftAnkleLat.abs();
    final rightRatio = rightKneeLat.abs() / rightAnkleLat.abs();
    return (leftRatio + rightRatio) / 2.0;
  }

  double? _kneeOverToeMetric3D(Pose pose, _BodyFrame frame) {
    final leftKnee = pose[PoseLandmarkType.leftKnee];
    final rightKnee = pose[PoseLandmarkType.rightKnee];
    final leftAnkle = pose[PoseLandmarkType.leftAnkle];
    final rightAnkle = pose[PoseLandmarkType.rightAnkle];
    final leftFoot = pose[PoseLandmarkType.leftFootIndex] ?? leftAnkle;
    final rightFoot = pose[PoseLandmarkType.rightFootIndex] ?? rightAnkle;
    if (!_isReliable(leftKnee) ||
        !_isReliable(rightKnee) ||
        !_isReliable(leftAnkle) ||
        !_isReliable(rightAnkle) ||
        !_isReliable(leftFoot) ||
        !_isReliable(rightFoot)) {
      return null;
    }

    // 只关心“膝盖超过脚尖多少”，没有超过时按 0 处理。
    final leftForward =
        ((_forwardCoord(leftKnee!, frame) - _forwardCoord(leftFoot!, frame))
                .clamp(0.0, double.infinity) as num)
            .toDouble();
    final rightForward =
        ((_forwardCoord(rightKnee!, frame) - _forwardCoord(rightFoot!, frame))
                .clamp(0.0, double.infinity) as num)
            .toDouble();

    final leftShank =
        _projectToSagittal(_vector3(leftAnkle!, leftKnee), frame).magnitude;
    final rightShank =
        _projectToSagittal(_vector3(rightAnkle!, rightKnee), frame).magnitude;
    if (leftShank < 1e-5 || rightShank < 1e-5) {
      return null;
    }

    // 用小腿长度归一化，减少不同身材带来的绝对距离差异。
    return ((leftForward / leftShank) + (rightForward / rightShank)) / 2.0;
  }

  /// 计算髋部偏离肩到踝参考线的程度。
  double? _bodyLineHipOffset(Pose pose, _BodyFrame frame) {
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
    return (h.up - expectedUp) / lineLen;
  }

  double? _elbowFlareMetric3D(Pose pose, _BodyFrame frame) {
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
    final leftUpper =
        _projectToFrontal(_vector3(leftShoulder!, leftElbow!), frame);
    final rightUpper =
        _projectToFrontal(_vector3(rightShoulder!, rightElbow!), frame);
    return _avg(
        _vectorAngle3(torso, leftUpper), _vectorAngle3(torso, rightUpper));
  }

  double? _neckNeutralAngle3D(Pose pose, _BodyFrame frame) {
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
    return _vectorAngle3(neck, torso);
  }

  /// 优先使用 3D 颈部角度，失败时退回到更稳定的 2D 角度。
  double? _neckNeutralAngleStable(Pose pose, _BodyFrame frame) {
    final angle3d = _neckNeutralAngle3D(pose, frame);
    if (angle3d != null && angle3d.isFinite && !angle3d.isNaN) {
      return angle3d;
    }

    final leftEar = pose[PoseLandmarkType.leftEar];
    final rightEar = pose[PoseLandmarkType.rightEar];
    final shoulderCenter = _midpoint(
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.rightShoulder],
    );
    final hipCenter = _midpoint(
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    );
    if (!_isReliableRep(leftEar) ||
        !_isReliableRep(rightEar) ||
        !_isReliableRep(shoulderCenter) ||
        !_isReliableRep(hipCenter)) {
      return null;
    }

    final earCenter = PoseLandmark(
      x: (leftEar!.x + rightEar!.x) / 2,
      y: (leftEar.y + rightEar.y) / 2,
      z: (leftEar.z + rightEar.z) / 2,
      likelihood: min(leftEar.likelihood, rightEar.likelihood),
    );
    final s = shoulderCenter!;
    final h = hipCenter!;
    final v1x = earCenter.x - s.x;
    final v1y = earCenter.y - s.y;
    final v2x = h.x - s.x;
    final v2y = h.y - s.y;
    final m1 = sqrt(v1x * v1x + v1y * v1y);
    final m2 = sqrt(v2x * v2x + v2y * v2y);
    if (m1 < 1e-5 || m2 < 1e-5) {
      return null;
    }
    final cosValue = (v1x * v2x + v1y * v2y) / (m1 * m2);
    return acos(cosValue.clamp(-1.0, 1.0)) * 180 / pi;
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

import 'dart:math';

import '../models/pose_landmark.dart';

class PoseMetricSnapshot {
  const PoseMetricSnapshot({
    required this.metrics,
    required this.inferredViewTag,
  });

  final Map<String, double> metrics;
  final String inferredViewTag;

  double? operator [](String key) => metrics[key];
}

class PoseMetricCalculator {
  const PoseMetricCalculator();

  static const double minLikelihood = 0.45;
  static const double minLikelihoodForRep = 0.30;

  PoseMetricSnapshot calculate(Pose pose) {
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

    final widthRatio = ((shoulderWidth + hipWidth) / 2.0) / torsoLength;
    if (widthRatio <= 0.32) {
      return 'side';
    }
    if (widthRatio <= 0.6) {
      return 'oblique';
    }
    return 'front';
  }

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

    final v1 = _vector3(b, a);
    final v2 = _vector3(b, c);
    final angle3d = _vectorAngle3(v1, v2);
    if (!angle3d.isNaN && angle3d.isFinite) {
      return angle3d;
    }

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

    return ((leftForward / leftShank) + (rightForward / rightShank)) / 2.0;
  }

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

    final neck =
        _projectToSagittal(_vector3(shoulderCenter!, earCenter!), frame);
    final torso =
        _projectToSagittal(_vector3(shoulderCenter, hipCenter!), frame);
    return _vectorAngle3(neck, torso);
  }

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
    return (180 - angleDeg).abs();
  }

  PoseLandmark? _midpoint(PoseLandmark? a, PoseLandmark? b) {
    if (!_isReliable(a) || !_isReliable(b)) {
      return null;
    }
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

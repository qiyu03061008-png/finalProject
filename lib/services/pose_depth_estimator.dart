import 'dart:math';

import '../models/pose_landmark.dart';
import '../models/user_profile.dart';

class PoseDepthEstimator {
  final Map<PoseLandmarkType, double> _smoothedDepth =
      <PoseLandmarkType, double>{};

  Pose estimateMonocular3D({
    required Pose pose2d,
    required UserProfile profile,
  }) {
    final approxDepth = _estimateGlobalMonocularDepth(pose2d, profile);

    final converted = <PoseLandmarkType, PoseLandmark>{};
    for (final entry in pose2d.landmarks.entries) {
      final depth = _smooth(
        entry.key,
        (approxDepth + _monocularDepthFallback(entry.value.z) * 0.15)
            .clamp(0.3, 4.0)
            .toDouble(),
      );
      converted[entry.key] = entry.value.copyWith(z: depth);
    }

    return pose2d.copyWith(
      landmarks: converted,
      depthMode: PoseDepthMode.monocular,
      source: pose2d.source, // 保留 blazepose / movenet 来源，方便后续分析器判断
      timestamp: pose2d.timestamp,
    );
  }

//计算肩部中心到髋部中心在画面里的像素距离，作为人体在画面中的大小参考
  double _torsoPixels(Pose pose) {
    final leftShoulder = pose[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose[PoseLandmarkType.rightShoulder];
    final leftHip = pose[PoseLandmarkType.leftHip];
    final rightHip = pose[PoseLandmarkType.rightHip];
    if (leftShoulder == null ||
        rightShoulder == null ||
        leftHip == null ||
        rightHip == null) {
      return 120;
    }
    final shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
    final shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
    final hipMidX = (leftHip.x + rightHip.x) / 2;
    final hipMidY = (leftHip.y + rightHip.y) / 2;
    final dx = shoulderMidX - hipMidX;
    final dy = shoulderMidY - hipMidY;
    return sqrt(dx * dx + dy * dy);
  }

  double _estimateGlobalMonocularDepth(Pose pose2d, UserProfile profile) {
    final torsoPixels = _torsoPixels(pose2d).clamp(35, 500).toDouble();
    final expectedTorsoMeters = profile.heightCm / 100 * 0.29;
    final focal = 700.0;
    return (expectedTorsoMeters * focal) / torsoPixels;
  }

  double _monocularDepthFallback(double modelZ) {
    return 1.2 + ((-modelZ / 180).clamp(-0.45, 0.45) as num).toDouble();
  }

  double _smooth(PoseLandmarkType type, double value) {
    final previous = _smoothedDepth[type];
    if (previous == null) {
      _smoothedDepth[type] = value;
      return value;
    }
    final smoothed = previous * 0.75 + value * 0.25;//75%的旧值+25%的新值
    _smoothedDepth[type] = smoothed;
    return smoothed;
  }
}

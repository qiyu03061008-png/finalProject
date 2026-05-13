enum PoseLandmarkType {
  nose,
  leftEyeInner,
  leftEye,
  leftEyeOuter,
  rightEyeInner,
  rightEye,
  rightEyeOuter,
  leftEar,
  rightEar,
  leftMouth,
  rightMouth,
  leftShoulder,
  rightShoulder,
  leftElbow,
  rightElbow,
  leftWrist,
  rightWrist,
  leftPinky,
  rightPinky,
  leftIndex,
  rightIndex,
  leftThumb,
  rightThumb,
  leftHip,
  rightHip,
  leftKnee,
  rightKnee,
  leftAnkle,
  rightAnkle,
  leftHeel,
  rightHeel,
  leftFootIndex,
  rightFootIndex,
}

class PoseLandmark {
  const PoseLandmark({
    required this.x,
    required this.y,
    required this.z,
    required this.likelihood,
  });

  final double x;
  final double y;
  final double z;
  final double likelihood;

  PoseLandmark copyWith({
    double? x,
    double? y,
    double? z,
    double? likelihood,
  }) {
    return PoseLandmark(
      x: x ?? this.x,
      y: y ?? this.y,
      z: z ?? this.z,
      likelihood: likelihood ?? this.likelihood,
    );
  }
}

class Pose {
  Pose({
    required this.landmarks,
    required this.timestamp,
    required this.source,
    this.depthMode = PoseDepthMode.monocular,
  });

  final Map<PoseLandmarkType, PoseLandmark> landmarks;
  final DateTime timestamp;
  final String source;
  final PoseDepthMode depthMode;

  PoseLandmark? operator [](PoseLandmarkType type) => landmarks[type];

  Pose copyWith({
    Map<PoseLandmarkType, PoseLandmark>? landmarks,
    DateTime? timestamp,
    String? source,
    PoseDepthMode? depthMode,
  }) {
    return Pose(
      landmarks: landmarks ?? this.landmarks,
      timestamp: timestamp ?? this.timestamp,
      source: source ?? this.source,
      depthMode: depthMode ?? this.depthMode,
    );
  }
}

enum PoseDepthMode {
  monocular,
}

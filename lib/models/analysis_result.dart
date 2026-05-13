enum ExerciseType {
  squat,
  pushup,
  plank,
}

extension ExerciseTypeX on ExerciseType {
  String get label {
    switch (this) {
      case ExerciseType.squat:
        return '深蹲';
      case ExerciseType.pushup:
        return '俯卧撑';
      case ExerciseType.plank:
        return '平板支撑';
    }
  }
}

enum PoseErrorType {
  kneeValgus,
  kneeOverToe,
  shallowSquat,
  torsoLeanForward,
  pushupDepthNotEnough,
  pushupHipSag,
  pushupHipPike,
  pushupElbowFlare,
  plankHipSag,
  plankHipPike,
  plankNeckNotNeutral,
}

class PoseIssue {
  const PoseIssue({
    required this.type,
    required this.message,
    required this.suggestion,
    required this.severity,
  });

  final PoseErrorType type;
  final String message;
  final String suggestion;
  final double severity;
}

class ExerciseAnalysisResult {
  const ExerciseAnalysisResult({
    required this.feedback,
    required this.score,
    required this.count,
    this.countDelta = 0,
    this.repJustCounted = false,
    this.repJustCountedClean = false,
    required this.issues,
    required this.metrics,
    required this.depthModeLabel,
  });

  final String feedback;
  final int score;
  final int count;
  final int countDelta;
  final bool repJustCounted;
  final bool repJustCountedClean;
  final List<PoseIssue> issues;
  final Map<String, double> metrics;
  final String depthModeLabel;
}

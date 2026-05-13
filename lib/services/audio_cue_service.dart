import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart';

import '../models/analysis_result.dart';

class AudioCueService {
  AudioCueService() {
    _player.setReleaseMode(ReleaseMode.stop);
  }

  final AudioPlayer _player = AudioPlayer();
  final Set<String> _missingAssets = <String>{};
  static const List<String> _supportedExtensions = <String>[
    '.wav',
    '.mp3',
  ];

  static const Map<PoseErrorType, String> _issueAssetMap =
      <PoseErrorType, String>{
        PoseErrorType.shallowSquat: 'audio/squat_depth_not_enough',
        PoseErrorType.torsoLeanForward: 'audio/torso_lean_forward',
        PoseErrorType.kneeValgus: 'audio/knees_caving_in',
        PoseErrorType.kneeOverToe: 'audio/knees_too_far_forward',
        PoseErrorType.pushupDepthNotEnough: 'audio/pushup_depth_not_enough',
        PoseErrorType.pushupHipSag: 'audio/hip_sag',
        PoseErrorType.pushupHipPike: 'audio/hip_too_high',
        PoseErrorType.pushupElbowFlare: 'audio/elbows_flared',
        PoseErrorType.plankHipSag: 'audio/plank_hip_sag',
        PoseErrorType.plankHipPike: 'audio/plank_hip_too_high',
        PoseErrorType.plankNeckNotNeutral: 'audio/neck_not_neutral',
      };

  static const Map<ExerciseType, String> _steadyAssetMap =
      <ExerciseType, String>{
        ExerciseType.squat: 'audio/squat_keep_steady',
        ExerciseType.pushup: 'audio/pushup_keep_steady',
        ExerciseType.plank: 'audio/plank_keep_steady',
      };

  Future<void> playForAnalysis({
    required ExerciseAnalysisResult analysis,
    required ExerciseType exerciseType,
  }) async {
    final assetPath = analysis.issues.isNotEmpty
        ? _issueAssetMap[analysis.issues.first.type]
        : _steadyAssetMap[exerciseType];
    if (assetPath == null) return;
    await _playAsset(assetPath);
  }

  Future<void> stop() => _player.stop();

  Future<void> dispose() async {
    await _player.dispose();
  }

  Future<void> _playAsset(String assetPath) async {
    for (final extension in _supportedExtensions) {
      final candidate = '$assetPath$extension';
      try {
        await _player.stop();
        await _player.play(AssetSource(candidate));
        return;
      } catch (_) {
        // Try the next supported extension before reporting a miss.
      }
    }
    if (_missingAssets.add(assetPath)) {
      debugPrint(
        'Audio cue asset missing or failed to play: assets/$assetPath.(wav|mp3)',
      );
    }
  }
}

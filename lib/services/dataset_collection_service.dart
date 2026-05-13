import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

import '../models/analysis_result.dart';
import '../models/pose_landmark.dart';
import '../models/user_profile.dart';

class DatasetValidationException implements Exception {
  const DatasetValidationException({
    required this.message,
    this.missingLandmarks = const <String>[],
  });

  final String message;
  final List<String> missingLandmarks;

  @override
  String toString() => message;
}

class DatasetProgress {
  const DatasetProgress({
    required this.targetSamples,
    required this.totalSamples,
    required this.completeSamples,
    required this.byExercise,
    required this.byQuality,
    required this.byView,
    required this.bySubject,
    required this.updatedAtIso,
  });

  factory DatasetProgress.empty({int targetSamples = 500}) {
    return DatasetProgress(
      targetSamples: targetSamples,
      totalSamples: 0,
      completeSamples: 0,
      byExercise: const <String, int>{},
      byQuality: const <String, int>{},
      byView: const <String, int>{},
      bySubject: const <String, int>{},
      updatedAtIso: DateTime.now().toIso8601String(),
    );
  }

  factory DatasetProgress.fromJson(Map<String, dynamic> json) {
    return DatasetProgress(
      targetSamples: (json['target_samples'] as num?)?.toInt() ?? 500,
      totalSamples: (json['total_samples'] as num?)?.toInt() ?? 0,
      completeSamples: (json['complete_samples'] as num?)?.toInt() ?? 0,
      byExercise: _toIntMap(json['by_exercise']),
      byQuality: _toIntMap(json['by_quality']),
      byView: _toIntMap(json['by_view']),
      bySubject: _toIntMap(json['by_subject']),
      updatedAtIso:
          json['updated_at'] as String? ?? DateTime.now().toIso8601String(),
    );
  }

  final int targetSamples;
  final int totalSamples;
  final int completeSamples;
  final Map<String, int> byExercise;
  final Map<String, int> byQuality;
  final Map<String, int> byView;
  final Map<String, int> bySubject;
  final String updatedAtIso;

  double get progressRatio {
    if (targetSamples <= 0) return 0;
    return (totalSamples / targetSamples).clamp(0, 1).toDouble();
  }

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'target_samples': targetSamples,
      'total_samples': totalSamples,
      'complete_samples': completeSamples,
      'by_exercise': byExercise,
      'by_quality': byQuality,
      'by_view': byView,
      'by_subject': bySubject,
      'updated_at': updatedAtIso,
    };
  }

  static Map<String, int> _toIntMap(dynamic value) {
    if (value is! Map) return <String, int>{};
    final out = <String, int>{};
    for (final entry in value.entries) {
      out[entry.key.toString()] = (entry.value as num?)?.toInt() ?? 0;
    }
    return out;
  }
}

class DatasetCollectionService {
  DatasetCollectionService({
    this.targetSamples = 500,
    Directory? datasetDirectory,
  })  : _uuid = const Uuid(),
        _datasetDirectoryOverride = datasetDirectory;

  final Uuid _uuid;
  final int targetSamples;
  final Directory? _datasetDirectoryOverride;
  Future<void> _statsWriteQueue = Future<void>.value();

  Future<String> appendSample({
    required Pose pose,
    required ExerciseType exerciseType,
    required String qualityTag,
    required String viewTag,
    required String subjectTag,
    required String annotatorId,
    required bool manualChecked,
    required UserProfile profile,
    required PoseDepthMode depthMode,
    String? primaryCameraName,
    Map<PoseLandmarkType, PoseLandmark>? correctedLandmarks,
    bool requireFullLandmarks = true,
    String cameraMode = 'single',
  }) async {
    final finalLandmarks = correctedLandmarks ?? pose.landmarks;
    if (requireFullLandmarks) {
      _validateLandmarksComplete(finalLandmarks);
    }

    final normalizedSubject =
        _normalizeTag(subjectTag, fallback: 'subject_unknown');
    final normalizedAnnotator =
        _normalizeTag(annotatorId, fallback: 'annotator_unknown');
    final normalizedCameraMode = _normalizeTag(cameraMode, fallback: 'single');
    _validateTagValue(
      field: 'quality_tag',
      value: qualityTag,
      allowed: const <String>{'standard', 'error'},
    );
    _validateTagValue(
      field: 'view_tag',
      value: viewTag,
      allowed: const <String>{'front', 'side', 'oblique'},
    );
    _validateTagValue(
      field: 'camera_mode',
      value: normalizedCameraMode,
      allowed: const <String>{'single'},
    );

    final leftCaptureTime = pose.timestamp;

    final file = await _resolveFile();
    final id = _uuid.v4();
    final fold = _crossValidationFold(normalizedSubject);
    final leftComplete =
        finalLandmarks.length == PoseLandmarkType.values.length;
    final isComplete = leftComplete;

    final payload = <String, dynamic>{
      'sample_id': id,
      'timestamp': pose.timestamp.toIso8601String(),
      'capture_time_left': leftCaptureTime.toIso8601String(),
      'exercise_type': exerciseType.name,
      'quality_tag': qualityTag,
      'view_tag': viewTag,
      'subject_tag': normalizedSubject,
      'annotation_protocol': 'v1.4_blazepose33_semiauto_single',
      'auto_label_source': 'MediaPipe_BlazePose',
      'manual_checked': manualChecked,
      'annotator_id': normalizedAnnotator,
      'annotation_workflow': 'single_annotator_optional_manual_review',
      'cross_validation_fold': fold,
      'depth_mode': depthMode.name,
      'camera_mode': normalizedCameraMode,
      'camera_primary': primaryCameraName,
      'smpl_shape_code': profile.toSmplShapeCode(),
      'user_profile': profile.toJson(),
      'landmarks_auto': _landmarksToJson(pose.landmarks),
      'landmarks_final': _landmarksToJson(finalLandmarks),
      'landmarks_complete': isComplete,
    };

    await file.writeAsString('${jsonEncode(payload)}\n', mode: FileMode.append);
    await _enqueueStatsUpdate(
      exerciseType: exerciseType.name,
      qualityTag: qualityTag,
      viewTag: viewTag,
      subjectTag: normalizedSubject,
      complete: isComplete,
    );
    return file.path;
  }

  Future<DatasetProgress> loadProgress() async {
    final file = await _statsFile();
    if (!await file.exists()) {
      return DatasetProgress.empty(targetSamples: targetSamples);
    }
    try {
      final raw = await file.readAsString();
      if (raw.trim().isEmpty) {
        return DatasetProgress.empty(targetSamples: targetSamples);
      }
      final decoded = jsonDecode(raw);
      if (decoded is! Map<String, dynamic>) {
        return DatasetProgress.empty(targetSamples: targetSamples);
      }
      final progress = DatasetProgress.fromJson(decoded);
      return DatasetProgress(
        targetSamples: targetSamples,
        totalSamples: progress.totalSamples,
        completeSamples: progress.completeSamples,
        byExercise: progress.byExercise,
        byQuality: progress.byQuality,
        byView: progress.byView,
        bySubject: progress.bySubject,
        updatedAtIso: progress.updatedAtIso,
      );
    } catch (_) {
      return DatasetProgress.empty(targetSamples: targetSamples);
    }
  }

  Future<String> exportGuideline() async {
    final dir = await _datasetDir();
    final file = File('${dir.path}/annotation_guideline.json');
    final guideline = <String, dynamic>{
      'version': '1.4',
      'keypoints': PoseLandmarkType.values.map((e) => e.name).toList(),
      'required_fields': <String>[
        'sample_id',
        'exercise_type',
        'quality_tag',
        'view_tag',
        'subject_tag',
        'annotator_id',
        'manual_checked',
        'cross_validation_fold',
        'camera_mode',
        'capture_time_left',
        'smpl_shape_code',
        'landmarks_auto',
        'landmarks_final',
        'landmarks_complete',
      ],
      'quality_labels': <String>['standard', 'error'],
      'view_labels': <String>['front', 'side', 'oblique'],
      'camera_modes': <String>['single'],
      'target_samples': targetSamples,
      'required_keypoint_count': PoseLandmarkType.values.length,
      'review_rule':
          'Single-camera collection with one annotator and optional manual review confirmation.',
      'validation_rule':
          'landmarks_final must include all 33 keypoints.',
    };
    await file.writeAsString(jsonEncode(guideline));
    return file.path;
  }

  Map<String, dynamic> _landmarksToJson(
      Map<PoseLandmarkType, PoseLandmark> lm) {
    final out = <String, dynamic>{};
    for (final entry in lm.entries) {
      out[entry.key.name] = <String, dynamic>{
        'x': entry.value.x,
        'y': entry.value.y,
        'z': entry.value.z,
        'likelihood': entry.value.likelihood,
      };
    }
    return out;
  }

  int _crossValidationFold(String id) {
    var hash = 0;
    for (final code in id.codeUnits) {
      hash = ((hash * 31) + code) & 0x7fffffff;
    }
    return hash % 5;
  }

  String _normalizeTag(String raw, {required String fallback}) {
    final normalized = raw.trim().toLowerCase().replaceAll(RegExp(r'\s+'), '_');
    if (normalized.isEmpty) return fallback;
    return normalized;
  }

  void _validateTagValue({
    required String field,
    required String value,
    required Set<String> allowed,
  }) {
    final normalized = value.trim().toLowerCase();
    if (allowed.contains(normalized)) return;
    throw DatasetValidationException(
      message:
          'Invalid $field: $value. Allowed values: ${allowed.join(', ')}.',
    );
  }

  void _validateLandmarksComplete(
      Map<PoseLandmarkType, PoseLandmark> landmarks) {
    final missing = <String>[];
    for (final type in PoseLandmarkType.values) {
      if (!landmarks.containsKey(type)) {
        missing.add(type.name);
      }
    }
    if (missing.isNotEmpty) {
      throw DatasetValidationException(
        message:
            'Missing ${missing.length} keypoints. Require all ${PoseLandmarkType.values.length} BlazePose landmarks.',
        missingLandmarks: missing,
      );
    }
  }

  Future<void> _updateStats({
    required String exerciseType,
    required String qualityTag,
    required String viewTag,
    required String subjectTag,
    required bool complete,
  }) async {
    final current = await loadProgress();
    final byExercise = Map<String, int>.from(current.byExercise);
    final byQuality = Map<String, int>.from(current.byQuality);
    final byView = Map<String, int>.from(current.byView);
    final bySubject = Map<String, int>.from(current.bySubject);

    _inc(byExercise, exerciseType);
    _inc(byQuality, qualityTag);
    _inc(byView, viewTag);
    _inc(bySubject, subjectTag);

    final next = DatasetProgress(
      targetSamples: targetSamples,
      totalSamples: current.totalSamples + 1,
      completeSamples: current.completeSamples + (complete ? 1 : 0),
      byExercise: byExercise,
      byQuality: byQuality,
      byView: byView,
      bySubject: bySubject,
      updatedAtIso: DateTime.now().toIso8601String(),
    );

    final file = await _statsFile();
    await file.writeAsString(jsonEncode(next.toJson()));
  }

  Future<void> _enqueueStatsUpdate({
    required String exerciseType,
    required String qualityTag,
    required String viewTag,
    required String subjectTag,
    required bool complete,
  }) {
    _statsWriteQueue = _statsWriteQueue.then((_) {
      return _updateStats(
        exerciseType: exerciseType,
        qualityTag: qualityTag,
        viewTag: viewTag,
        subjectTag: subjectTag,
        complete: complete,
      );
    });
    return _statsWriteQueue;
  }

  void _inc(Map<String, int> map, String key) {
    map[key] = (map[key] ?? 0) + 1;
  }

  Future<File> _resolveFile() async {
    final dir = await _datasetDir();
    final date = DateTime.now();
    final fileName =
        'fitness_dataset_${date.year}${date.month.toString().padLeft(2, '0')}${date.day.toString().padLeft(2, '0')}.jsonl';
    final file = File('${dir.path}/$fileName');
    if (!await file.exists()) {
      await file.create(recursive: true);
    }
    return file;
  }

  Future<File> _statsFile() async {
    final dir = await _datasetDir();
    final file = File('${dir.path}/dataset_stats.json');
    if (!await file.exists()) {
      await file.create(recursive: true);
    }
    return file;
  }

  Future<Directory> _datasetDir() async {
    if (_datasetDirectoryOverride != null) {
      if (!await _datasetDirectoryOverride!.exists()) {
        await _datasetDirectoryOverride!.create(recursive: true);
      }
      return _datasetDirectoryOverride!;
    }
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/dataset');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }
}

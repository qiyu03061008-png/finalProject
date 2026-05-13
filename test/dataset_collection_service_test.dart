import 'dart:convert';
import 'dart:io';

import 'package:fitness_pose_app/models/analysis_result.dart';
import 'package:fitness_pose_app/models/pose_landmark.dart';
import 'package:fitness_pose_app/models/user_profile.dart';
import 'package:fitness_pose_app/services/dataset_collection_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late Directory tempDir;
  late DatasetCollectionService service;

  setUp(() async {
    tempDir = await Directory.systemTemp.createTemp('dataset_service_test_');
    service = DatasetCollectionService(datasetDirectory: tempDir);
  });

  tearDown(() async {
    if (await tempDir.exists()) {
      await tempDir.delete(recursive: true);
    }
  });

  test('appendSample writes normalized single-camera payload', () async {
    final filePath = await service.appendSample(
      pose: _buildPose(),
      exerciseType: ExerciseType.squat,
      qualityTag: 'standard',
      viewTag: 'front',
      subjectTag: ' Subject 01 ',
      annotatorId: ' Annotator A ',
      manualChecked: true,
      profile: UserProfile.defaultProfile(),
      depthMode: PoseDepthMode.monocular,
      primaryCameraName: 'front-camera',
      cameraMode: 'single',
    );

    final file = File(filePath);
    expect(await file.exists(), isTrue);

    final lines = await file.readAsLines();
    expect(lines, hasLength(1));
    final payload = jsonDecode(lines.single) as Map<String, dynamic>;
    expect(payload['annotation_protocol'], 'v1.4_blazepose33_semiauto_single');
    expect(payload['annotation_workflow'],
        'single_annotator_optional_manual_review');
    expect(payload['subject_tag'], 'subject_01');
    expect(payload['annotator_id'], 'annotator_a');
    expect(payload['camera_mode'], 'single');
    expect(payload['depth_mode'], 'monocular');
    expect(payload['camera_primary'], 'front-camera');
    expect(payload['landmarks_complete'], isTrue);
    expect((payload['landmarks_final'] as Map<String, dynamic>).length,
        PoseLandmarkType.values.length);
  });

  test('appendSample rejects invalid tags', () async {
    expect(
      () => service.appendSample(
        pose: _buildPose(),
        exerciseType: ExerciseType.pushup,
        qualityTag: 'bad_quality',
        viewTag: 'front',
        subjectTag: 'subject',
        annotatorId: 'annotator',
        manualChecked: false,
        profile: UserProfile.defaultProfile(),
        depthMode: PoseDepthMode.monocular,
      ),
      throwsA(isA<DatasetValidationException>()),
    );

    expect(
      () => service.appendSample(
        pose: _buildPose(),
        exerciseType: ExerciseType.pushup,
        qualityTag: 'standard',
        viewTag: 'rear',
        subjectTag: 'subject',
        annotatorId: 'annotator',
        manualChecked: false,
        profile: UserProfile.defaultProfile(),
        depthMode: PoseDepthMode.monocular,
      ),
      throwsA(isA<DatasetValidationException>()),
    );
  });

  test('appendSample rejects incomplete landmarks when full set is required',
      () async {
    final landmarks = Map<PoseLandmarkType, PoseLandmark>.from(_buildPose().landmarks)
      ..remove(PoseLandmarkType.leftAnkle);
    final incompletePose = Pose(
      landmarks: landmarks,
      timestamp: DateTime.parse('2026-05-07T12:00:00.000Z'),
      source: 'test',
    );

    expect(
      () => service.appendSample(
        pose: incompletePose,
        exerciseType: ExerciseType.plank,
        qualityTag: 'standard',
        viewTag: 'front',
        subjectTag: 'subject',
        annotatorId: 'annotator',
        manualChecked: false,
        profile: UserProfile.defaultProfile(),
        depthMode: PoseDepthMode.monocular,
      ),
      throwsA(isA<DatasetValidationException>()),
    );
  });

  test('exportGuideline matches single-camera workflow', () async {
    final filePath = await service.exportGuideline();
    final payload =
        jsonDecode(await File(filePath).readAsString()) as Map<String, dynamic>;

    expect(payload['version'], '1.4');
    expect(payload['camera_modes'], <String>['single']);
    expect(
      payload['review_rule'],
      'Single-camera collection with one annotator and optional manual review confirmation.',
    );
    expect(
      payload['validation_rule'],
      'landmarks_final must include all 33 keypoints.',
    );
  });
}

Pose _buildPose() {
  final landmarks = <PoseLandmarkType, PoseLandmark>{};
  for (var i = 0; i < PoseLandmarkType.values.length; i += 1) {
    final type = PoseLandmarkType.values[i];
    landmarks[type] = PoseLandmark(
      x: i * 3.0 + 10,
      y: i * 2.0 + 20,
      z: -0.1 * i,
      likelihood: 0.95,
    );
  }
  return Pose(
    landmarks: landmarks,
    timestamp: DateTime.parse('2026-05-07T12:00:00.000Z'),
    source: 'test',
  );
}

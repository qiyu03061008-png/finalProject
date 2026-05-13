import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart'
    as mlkit;

import '../models/pose_landmark.dart';

class BlazePoseDetector {
  BlazePoseDetector({
    bool strictHumanValidation = true,
  }) : _strictHumanValidation = strictHumanValidation,
       _detector = mlkit.PoseDetector(
         options: mlkit.PoseDetectorOptions(
           mode: mlkit.PoseDetectionMode.stream,
         ),
       );

  final mlkit.PoseDetector _detector;
  final bool _strictHumanValidation;
  bool _closed = false;

  static const double _minReliableLikelihood = 0.55;
  static const double _minVisibleLikelihood = 0.40;

  static final Map<String, PoseLandmarkType> _landmarkMap =
      <String, PoseLandmarkType>{
        'nose': PoseLandmarkType.nose,
        'leftEyeInner': PoseLandmarkType.leftEyeInner,
        'leftEye': PoseLandmarkType.leftEye,
        'leftEyeOuter': PoseLandmarkType.leftEyeOuter,
        'rightEyeInner': PoseLandmarkType.rightEyeInner,
        'rightEye': PoseLandmarkType.rightEye,
        'rightEyeOuter': PoseLandmarkType.rightEyeOuter,
        'leftEar': PoseLandmarkType.leftEar,
        'rightEar': PoseLandmarkType.rightEar,
        'leftMouth': PoseLandmarkType.leftMouth,
        'rightMouth': PoseLandmarkType.rightMouth,
        'mouthLeft': PoseLandmarkType.leftMouth,
        'mouthRight': PoseLandmarkType.rightMouth,
        'leftShoulder': PoseLandmarkType.leftShoulder,
        'rightShoulder': PoseLandmarkType.rightShoulder,
        'leftElbow': PoseLandmarkType.leftElbow,
        'rightElbow': PoseLandmarkType.rightElbow,
        'leftWrist': PoseLandmarkType.leftWrist,
        'rightWrist': PoseLandmarkType.rightWrist,
        'leftPinky': PoseLandmarkType.leftPinky,
        'rightPinky': PoseLandmarkType.rightPinky,
        'leftIndex': PoseLandmarkType.leftIndex,
        'rightIndex': PoseLandmarkType.rightIndex,
        'leftThumb': PoseLandmarkType.leftThumb,
        'rightThumb': PoseLandmarkType.rightThumb,
        'leftHip': PoseLandmarkType.leftHip,
        'rightHip': PoseLandmarkType.rightHip,
        'leftKnee': PoseLandmarkType.leftKnee,
        'rightKnee': PoseLandmarkType.rightKnee,
        'leftAnkle': PoseLandmarkType.leftAnkle,
        'rightAnkle': PoseLandmarkType.rightAnkle,
        'leftHeel': PoseLandmarkType.leftHeel,
        'rightHeel': PoseLandmarkType.rightHeel,
        'leftFootIndex': PoseLandmarkType.leftFootIndex,
        'rightFootIndex': PoseLandmarkType.rightFootIndex,
      };

  Future<Pose?> detectFromCameraImage({
    required CameraImage image,
    required CameraDescription camera,
    DateTime? captureTime,
  }) async {
    if (_closed) return null;

    final inputImage = _toInputImage(image, camera);
    if (inputImage == null) return null;

    final poses = await _detector.processImage(inputImage);
    if (poses.isEmpty) return null;

    final mappedPose = _mapPose(poses.first, captureTime ?? DateTime.now());
    if (_strictHumanValidation && !_isValidHumanPose(mappedPose)) {
      return null;
    }
    return mappedPose;
  }

  void close() {
    if (_closed) return;
    _detector.close();
    _closed = true;
  }

  mlkit.InputImage? _toInputImage(
    CameraImage image,
    CameraDescription camera,
  ) {
    final rotation =
        mlkit.InputImageRotationValue.fromRawValue(camera.sensorOrientation);
    final rawFormat =
        mlkit.InputImageFormatValue.fromRawValue(image.format.raw);
    if (rotation == null) return null;
    if (image.planes.isEmpty) return null;

    late final Uint8List bytes;
    late final mlkit.InputImageFormat format;
    late final int bytesPerRow;

    if (image.planes.length == 1) {
      if (rawFormat == null) return null;
      bytes = image.planes.first.bytes;
      format = rawFormat;
      bytesPerRow = image.planes.first.bytesPerRow;
    } else if (rawFormat == mlkit.InputImageFormat.yuv_420_888 &&
        image.planes.length >= 3) {
      bytes = _yuv420ToNv21(image);
      format = mlkit.InputImageFormat.nv21;
      bytesPerRow = image.width;
    } else {
      if (rawFormat == null) return null;
      bytes = _concatenatePlanes(image.planes);
      format = rawFormat;
      bytesPerRow = image.planes.first.bytesPerRow;
    }

    if (bytes.isEmpty) return null;

    final metadata = mlkit.InputImageMetadata(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      rotation: rotation,
      format: format,
      bytesPerRow: bytesPerRow,
    );

    return mlkit.InputImage.fromBytes(bytes: bytes, metadata: metadata);
  }

  static Uint8List _yuv420ToNv21(CameraImage image) {
    final width = image.width;
    final height = image.height;
    if (image.planes.length < 3) {
      return Uint8List(0);
    }

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final ySize = width * height;
    final uvWidth = width ~/ 2;
    final uvHeight = height ~/ 2;
    final out = Uint8List(ySize + uvWidth * uvHeight * 2);

    int outIndex = 0;

    final yBytes = yPlane.bytes;
    final yRowStride = yPlane.bytesPerRow;
    final yPixelStride = yPlane.bytesPerPixel ?? 1;
    for (var row = 0; row < height; row++) {
      final rowStart = row * yRowStride;
      for (var col = 0; col < width; col++) {
        final srcIndex = rowStart + col * yPixelStride;
        out[outIndex++] = srcIndex < yBytes.length ? yBytes[srcIndex] : 0;
      }
    }

    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;
    final uRowStride = uPlane.bytesPerRow;
    final vRowStride = vPlane.bytesPerRow;
    final uPixelStride = uPlane.bytesPerPixel ?? 1;
    final vPixelStride = vPlane.bytesPerPixel ?? 1;
    for (var row = 0; row < uvHeight; row++) {
      final uRowStart = row * uRowStride;
      final vRowStart = row * vRowStride;
      for (var col = 0; col < uvWidth; col++) {
        final vIndex = vRowStart + col * vPixelStride;
        final uIndex = uRowStart + col * uPixelStride;
        out[outIndex++] = vIndex < vBytes.length ? vBytes[vIndex] : 0;
        out[outIndex++] = uIndex < uBytes.length ? uBytes[uIndex] : 0;
      }
    }

    return out;
  }

  static Uint8List _concatenatePlanes(List<Plane> planes) {
    int totalSize = 0;
    for (final p in planes) {
      totalSize += p.bytes.length;
    }
    final out = Uint8List(totalSize);
    int offset = 0;
    for (final p in planes) {
      out.setRange(offset, offset + p.bytes.length, p.bytes);
      offset += p.bytes.length;
    }
    return out;
  }

  Pose _mapPose(mlkit.Pose pose, DateTime timestamp) {
    final mapped = <PoseLandmarkType, PoseLandmark>{};
    for (final entry in pose.landmarks.entries) {
      final type = _landmarkMap[entry.key.name];
      if (type == null) continue;
      mapped[type] = PoseLandmark(
        x: entry.value.x,
        y: entry.value.y,
        z: entry.value.z,
        likelihood: entry.value.likelihood,
      );
    }

    return Pose(
      landmarks: mapped,
      timestamp: timestamp,
      source: 'blazepose',
      depthMode: PoseDepthMode.monocular,
    );
  }

  bool _isValidHumanPose(Pose pose) {
    if (pose.landmarks.isEmpty) return false;

    var reliableCount = 0;
    for (final landmark in pose.landmarks.values) {
      if (landmark.likelihood >= _minReliableLikelihood) {
        reliableCount += 1;
      }
    }
    if (reliableCount < 12) return false;

    final leftShoulder = pose[PoseLandmarkType.leftShoulder];
    final rightShoulder = pose[PoseLandmarkType.rightShoulder];
    final leftHip = pose[PoseLandmarkType.leftHip];
    final rightHip = pose[PoseLandmarkType.rightHip];

    final leftTorsoVisible = _isVisible(leftShoulder) && _isVisible(leftHip);
    final rightTorsoVisible = _isVisible(rightShoulder) && _isVisible(rightHip);
    if (!leftTorsoVisible && !rightTorsoVisible) {
      return false;
    }

    final torsoHeight = _torsoHeight(
      leftShoulder: leftShoulder,
      rightShoulder: rightShoulder,
      leftHip: leftHip,
      rightHip: rightHip,
    );
    if (torsoHeight < 25) return false;

    if (_isVisible(leftShoulder) && _isVisible(rightShoulder)) {
      final shoulderSpan = (leftShoulder!.x - rightShoulder!.x).abs();
      if (shoulderSpan / torsoHeight < 0.12) return false;

      final shoulderTilt = (leftShoulder.y - rightShoulder.y).abs();
      if (shoulderTilt > torsoHeight * 0.75) return false;
    }

    if (_isVisible(leftHip) && _isVisible(rightHip)) {
      final hipSpan = (leftHip!.x - rightHip!.x).abs();
      if (hipSpan / torsoHeight < 0.10) return false;

      final hipTilt = (leftHip.y - rightHip.y).abs();
      if (hipTilt > torsoHeight * 0.75) return false;
    }

    final reliableArmCount = [
      pose[PoseLandmarkType.leftElbow],
      pose[PoseLandmarkType.rightElbow],
      pose[PoseLandmarkType.leftWrist],
      pose[PoseLandmarkType.rightWrist],
    ].where(_isReliable).length;

    final reliableLegCount = [
      pose[PoseLandmarkType.leftKnee],
      pose[PoseLandmarkType.rightKnee],
      pose[PoseLandmarkType.leftAnkle],
      pose[PoseLandmarkType.rightAnkle],
    ].where(_isReliable).length;

    if (reliableArmCount < 1) return false;
    if (reliableLegCount < 1) return false;

    if (leftTorsoVisible && leftShoulder!.y >= leftHip!.y) return false;
    if (rightTorsoVisible && rightShoulder!.y >= rightHip!.y) return false;

    return true;
  }

  bool _isReliable(PoseLandmark? point) {
    return point != null && point.likelihood >= _minReliableLikelihood;
  }

  bool _isVisible(PoseLandmark? point) {
    return point != null && point.likelihood >= _minVisibleLikelihood;
  }

  double _torsoHeight({
    required PoseLandmark? leftShoulder,
    required PoseLandmark? rightShoulder,
    required PoseLandmark? leftHip,
    required PoseLandmark? rightHip,
  }) {
    if (_isVisible(leftShoulder) &&
        _isVisible(rightShoulder) &&
        _isVisible(leftHip) &&
        _isVisible(rightHip)) {
      return (((leftShoulder!.y + rightShoulder!.y) / 2) -
              ((leftHip!.y + rightHip!.y) / 2))
          .abs();
    }
    if (_isVisible(leftShoulder) && _isVisible(leftHip)) {
      return (leftHip!.y - leftShoulder!.y).abs();
    }
    if (_isVisible(rightShoulder) && _isVisible(rightHip)) {
      return (rightHip!.y - rightShoulder!.y).abs();
    }
    return 0;
  }
}

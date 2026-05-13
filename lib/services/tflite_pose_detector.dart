import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tflite;

import '../models/pose_landmark.dart';

/// MoveNet single-person detector based on TFLite inference.
///
/// This version prefers broad device compatibility over aggressive
/// optimization. Some lower-level buffer optimizations were causing
/// invalid inference output on certain Android devices.
class TFLitePoseDetector {
  tflite.Interpreter? _interpreter;
  bool _initialized = false;
  int _inputHeight = 192;
  int _inputWidth = 192;
  tflite.TensorType _inputType = tflite.TensorType.uint8;
  tflite.TensorType _outputType = tflite.TensorType.float32;
  List<int> _outputShape = const <int>[1, 1, 17, 3];
  Object? _inputBuffer;
  Object? _outputBuffer;

  static const double _minReliableScore = 0.20;

  static const Map<int, PoseLandmarkType> _movenetIndexMap =
      <int, PoseLandmarkType>{
        0: PoseLandmarkType.nose,
        1: PoseLandmarkType.leftEye,
        2: PoseLandmarkType.rightEye,
        3: PoseLandmarkType.leftEar,
        4: PoseLandmarkType.rightEar,
        5: PoseLandmarkType.leftShoulder,
        6: PoseLandmarkType.rightShoulder,
        7: PoseLandmarkType.leftElbow,
        8: PoseLandmarkType.rightElbow,
        9: PoseLandmarkType.leftWrist,
        10: PoseLandmarkType.rightWrist,
        11: PoseLandmarkType.leftHip,
        12: PoseLandmarkType.rightHip,
        13: PoseLandmarkType.leftKnee,
        14: PoseLandmarkType.rightKnee,
        15: PoseLandmarkType.leftAnkle,
        16: PoseLandmarkType.rightAnkle,
      };

  bool get isReady => _initialized && _interpreter != null;

  Future<void> initialize() async {
    if (_initialized) return;
    try {
      final interpreter = await tflite.Interpreter.fromAsset(
        'assets/models/movenet_lightning.tflite',
        options: tflite.InterpreterOptions()..threads = 1,
      );
      final inputTensor = interpreter.getInputTensor(0);
      final inputShape = inputTensor.shape;
      if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[3] != 3) {
        interpreter.close();
        throw StateError('Unexpected MoveNet input tensor shape: $inputShape');
      }

      _inputHeight = inputShape[1];
      _inputWidth = inputShape[2];
      _inputType = inputTensor.type;

      final outputTensor = interpreter.getOutputTensor(0);
      _outputType = outputTensor.type;
      _outputShape = outputTensor.shape;
      _inputBuffer = _createInputBuffer();
      _outputBuffer = _createOutputBuffer();

      _interpreter = interpreter;
      _initialized = true;
    } catch (_) {
      _interpreter?.close();
      _interpreter = null;
      _initialized = false;
      _inputBuffer = null;
      _outputBuffer = null;
    }
  }

  Future<Pose?> detectPoseFromCamera(
    CameraImage cameraImage,
    CameraDescription camera,
  ) async {
    if (!isReady) return null;

    final rawImage = _cameraImageToRgb(cameraImage);
    if (rawImage == null) return null;
    final upright =
        _rotateToDisplayOrientation(rawImage, camera.sensorOrientation);
    return _detectFromImage(
      upright,
      source: 'movenet_camera',
      timestamp: DateTime.now(),
    );
  }

  Future<Pose?> detectPose(Uint8List imageBytes, int width, int height) async {
    if (!isReady) return null;
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) return null;
    return _detectFromImage(
      decoded,
      source: 'movenet_image',
      timestamp: DateTime.now(),
    );
  }

  Pose? _detectFromImage(
    img.Image image, {
    required String source,
    required DateTime timestamp,
  }) {
    final interpreter = _interpreter;
    if (!_initialized || interpreter == null) return null;

    final prep = _letterboxToInput(image);
    final output = _outputBuffer ??= _createOutputBuffer();
    _resetOutputBuffer(output);
    interpreter.run(prep.input, output);

    final keypoints = _extractKeypoints(output);
    if (keypoints.length < 17) return null;

    return _mapKeypointsToPose(
      keypoints: keypoints,
      sourceWidth: image.width,
      sourceHeight: image.height,
      scale: prep.scale,
      padX: prep.padX,
      padY: prep.padY,
      source: source,
      timestamp: timestamp,
    );
  }

  Pose? _mapKeypointsToPose({
    required List<(double, double, double)> keypoints,
    required int sourceWidth,
    required int sourceHeight,
    required double scale,
    required double padX,
    required double padY,
    required String source,
    required DateTime timestamp,
  }) {
    final mapped = <PoseLandmarkType, PoseLandmark>{};
    for (var i = 0; i < keypoints.length; i++) {
      final type = _movenetIndexMap[i];
      if (type == null) continue;

      final kp = keypoints[i];
      final score = kp.$3;
      if (score < _minReliableScore) continue;

      final yInInput = kp.$1 * _inputHeight;
      final xInInput = kp.$2 * _inputWidth;
      final y = ((yInInput - padY) / scale)
          .clamp(0.0, sourceHeight.toDouble())
          .toDouble();
      final x = ((xInInput - padX) / scale)
          .clamp(0.0, sourceWidth.toDouble())
          .toDouble();

      mapped[type] = PoseLandmark(
        x: x,
        y: y,
        z: 0,
        likelihood: score.clamp(0.0, 1.0).toDouble(),
      );
    }

    _addDerivedFaceLandmarks(mapped);
    if (!_looksLikeHuman(mapped)) return null;

    return Pose(
      landmarks: mapped,
      timestamp: timestamp,
      source: source,
      depthMode: PoseDepthMode.monocular,
    );
  }

  _LetterboxPrep _letterboxToInput(img.Image image) {
    final scale =
        math.min(_inputWidth / image.width, _inputHeight / image.height);
    final resizedWidth = math.max(1, (image.width * scale).round());
    final resizedHeight = math.max(1, (image.height * scale).round());
    final resized = img.copyResize(
      image,
      width: resizedWidth,
      height: resizedHeight,
      interpolation: img.Interpolation.average,
    );

    final padX = (_inputWidth - resizedWidth) / 2.0;
    final padY = (_inputHeight - resizedHeight) / 2.0;
    final bg = img.Image(width: _inputWidth, height: _inputHeight);
    img.fill(bg, color: img.ColorRgb8(0, 0, 0));
    img.compositeImage(bg, resized, dstX: padX.round(), dstY: padY.round());

    final input = _inputBuffer ??= _createInputBuffer();
    final frame = (input as List).first as List;
    for (var y = 0; y < _inputHeight; y++) {
      final row = frame[y] as List;
      for (var x = 0; x < _inputWidth; x++) {
        final pixel = row[x] as List;
        final p = bg.getPixel(x, y);
        if (_inputType == tflite.TensorType.uint8) {
          pixel[0] = p.r.toInt();
          pixel[1] = p.g.toInt();
          pixel[2] = p.b.toInt();
        } else {
          pixel[0] = p.r / 255.0;
          pixel[1] = p.g / 255.0;
          pixel[2] = p.b / 255.0;
        }
      }
    }
    return _LetterboxPrep(input, scale, padX, padY);
  }

  Object _createInputBuffer() {
    if (_inputType == tflite.TensorType.uint8) {
      return List.generate(
        1,
        (_) => List.generate(
          _inputHeight,
          (_) => List.generate(
            _inputWidth,
            (_) => <int>[0, 0, 0],
            growable: false,
          ),
          growable: false,
        ),
        growable: false,
      );
    }

    return List.generate(
      1,
      (_) => List.generate(
        _inputHeight,
        (_) => List.generate(
          _inputWidth,
          (_) => <double>[0, 0, 0],
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );
  }

  Object _createOutputBuffer() {
    final is173 =
        _outputShape.length == 3 &&
        _outputShape[0] == 1 &&
        _outputShape[1] == 17 &&
        _outputShape[2] == 3;

    if (is173) {
      if (_outputType == tflite.TensorType.float32) {
        return List.generate(
          1,
          (_) => List.generate(
            17,
            (_) => List<double>.filled(3, 0.0),
            growable: false,
          ),
          growable: false,
        );
      }
      return List.generate(
        1,
        (_) => List.generate(
          17,
          (_) => List<int>.filled(3, 0),
          growable: false,
        ),
        growable: false,
      );
    }

    if (_outputType == tflite.TensorType.float32) {
      return List.generate(
        1,
        (_) => List.generate(
          1,
          (_) => List.generate(
            17,
            (_) => List<double>.filled(3, 0.0),
            growable: false,
          ),
          growable: false,
        ),
        growable: false,
      );
    }
    return List.generate(
      1,
      (_) => List.generate(
        1,
        (_) => List.generate(
          17,
          (_) => List<int>.filled(3, 0),
          growable: false,
        ),
        growable: false,
      ),
      growable: false,
    );
  }

  void _resetOutputBuffer(Object output) {
    if (output is! List) return;
    _resetNestedList(output);
  }

  void _resetNestedList(List values) {
    for (var i = 0; i < values.length; i++) {
      final value = values[i];
      if (value is List) {
        _resetNestedList(value);
      } else if (value is double) {
        values[i] = 0.0;
      } else if (value is int) {
        values[i] = 0;
      }
    }
  }

  List<(double, double, double)> _extractKeypoints(Object output) {
    final result = <(double, double, double)>[];
    if (output is! List || output.isEmpty) return result;

    dynamic level = output.first;
    if (level is List && level.isNotEmpty && level.first is List) {
      final firstInner = level.first;
      if (firstInner is List && firstInner.length == 17) {
        level = firstInner;
      }
    }

    if (level is! List) return result;
    for (final row in level) {
      if (row is List && row.length >= 3) {
        result.add((_asDouble(row[0]), _asDouble(row[1]), _asDouble(row[2])));
      }
    }
    return result;
  }

  double _asDouble(Object value) {
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is num) return value.toDouble();
    return 0.0;
  }

  img.Image? _cameraImageToRgb(CameraImage image) {
    if (image.format.group == ImageFormatGroup.bgra8888) {
      return _bgra8888ToImage(image);
    }
    if (image.format.group == ImageFormatGroup.yuv420 ||
        image.planes.length >= 3) {
      return _yuv420ToImage(image);
    }
    return null;
  }

  img.Image _rotateToDisplayOrientation(img.Image src, int sensorOrientation) {
    switch (sensorOrientation) {
      case 90:
        return img.copyRotate(src, angle: 90);
      case 180:
        return img.copyRotate(src, angle: 180);
      case 270:
        return img.copyRotate(src, angle: -90);
      default:
        return src;
    }
  }

  img.Image? _bgra8888ToImage(CameraImage image) {
    if (image.planes.isEmpty) return null;
    final plane = image.planes.first;
    final bytes = plane.bytes;
    final rowStride = plane.bytesPerRow;
    final out = img.Image(width: image.width, height: image.height);

    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final offset = y * rowStride + x * 4;
        if (offset + 3 >= bytes.length) {
          out.setPixelRgb(x, y, 0, 0, 0);
          continue;
        }
        final b = bytes[offset];
        final g = bytes[offset + 1];
        final r = bytes[offset + 2];
        out.setPixelRgb(x, y, r, g, b);
      }
    }
    return out;
  }

  img.Image? _yuv420ToImage(CameraImage image) {
    if (image.planes.length < 3) return null;
    final width = image.width;
    final height = image.height;
    final out = img.Image(width: width, height: height);

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];
    final yBytes = yPlane.bytes;
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;

    final yRowStride = yPlane.bytesPerRow;
    final yPixelStride = yPlane.bytesPerPixel ?? 1;
    final uRowStride = uPlane.bytesPerRow;
    final uPixelStride = uPlane.bytesPerPixel ?? 1;
    final vRowStride = vPlane.bytesPerRow;
    final vPixelStride = vPlane.bytesPerPixel ?? 1;

    for (var y = 0; y < height; y++) {
      final uvY = y ~/ 2;
      for (var x = 0; x < width; x++) {
        final uvX = x ~/ 2;
        final yIndex = y * yRowStride + x * yPixelStride;
        final uIndex = uvY * uRowStride + uvX * uPixelStride;
        final vIndex = uvY * vRowStride + uvX * vPixelStride;
        if (yIndex >= yBytes.length ||
            uIndex >= uBytes.length ||
            vIndex >= vBytes.length) {
          out.setPixelRgb(x, y, 0, 0, 0);
          continue;
        }
        final yp = yBytes[yIndex].toDouble();
        final up = uBytes[uIndex].toDouble();
        final vp = vBytes[vIndex].toDouble();
        final r = (yp + 1.402 * (vp - 128)).round().clamp(0, 255);
        final g = (yp - 0.344136 * (up - 128) - 0.714136 * (vp - 128))
            .round()
            .clamp(0, 255);
        final b = (yp + 1.772 * (up - 128)).round().clamp(0, 255);
        out.setPixelRgb(x, y, r, g, b);
      }
    }
    return out;
  }

  void _addDerivedFaceLandmarks(Map<PoseLandmarkType, PoseLandmark> mapped) {
    final nose = mapped[PoseLandmarkType.nose];
    final leftEye = mapped[PoseLandmarkType.leftEye];
    final rightEye = mapped[PoseLandmarkType.rightEye];
    final leftEar = mapped[PoseLandmarkType.leftEar];
    final rightEar = mapped[PoseLandmarkType.rightEar];
    if (leftEye != null) {
      mapped[PoseLandmarkType.leftEyeInner] = leftEye;
      mapped[PoseLandmarkType.leftEyeOuter] = leftEye;
    }
    if (rightEye != null) {
      mapped[PoseLandmarkType.rightEyeInner] = rightEye;
      mapped[PoseLandmarkType.rightEyeOuter] = rightEye;
    }
    if (nose != null && leftEar != null && rightEar != null) {
      final mouthY = nose.y + ((leftEar.y + rightEar.y) / 2 - nose.y) * 0.55;
      final spread = (rightEar.x - leftEar.x).abs() * 0.18;
      final likelihood = math.min(
            nose.likelihood,
            math.min(leftEar.likelihood, rightEar.likelihood),
          ) *
          0.8;
      mapped[PoseLandmarkType.leftMouth] = PoseLandmark(
        x: nose.x - spread,
        y: mouthY,
        z: 0,
        likelihood: likelihood,
      );
      mapped[PoseLandmarkType.rightMouth] = PoseLandmark(
        x: nose.x + spread,
        y: mouthY,
        z: 0,
        likelihood: likelihood,
      );
    }
  }

  bool _looksLikeHuman(Map<PoseLandmarkType, PoseLandmark> mapped) {
    final leftShoulder = mapped[PoseLandmarkType.leftShoulder];
    final rightShoulder = mapped[PoseLandmarkType.rightShoulder];
    final leftHip = mapped[PoseLandmarkType.leftHip];
    final rightHip = mapped[PoseLandmarkType.rightHip];
    if (leftShoulder == null ||
        rightShoulder == null ||
        leftHip == null ||
        rightHip == null) {
      return false;
    }
    final torsoReliableCount = <double>[
      leftShoulder.likelihood,
      rightShoulder.likelihood,
      leftHip.likelihood,
      rightHip.likelihood,
    ].where((s) => s >= _minReliableScore).length;
    return torsoReliableCount >= 3;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _initialized = false;
    _inputBuffer = null;
    _outputBuffer = null;
  }
}

class _LetterboxPrep {
  const _LetterboxPrep(this.input, this.scale, this.padX, this.padY);

  final Object input;
  final double scale;
  final double padX;
  final double padY;
}

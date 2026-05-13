import 'package:flutter/material.dart';

import '../models/pose_landmark.dart';

class PosePainter extends CustomPainter {
  PosePainter({
    required this.pose,
    required this.imageSize,
    required this.canvasSize,
    this.mirrorX = false,
  });

  final Pose pose;
  final Size imageSize;
  final Size canvasSize;
  final bool mirrorX;

  static final List<List<PoseLandmarkType>> _connections = <List<PoseLandmarkType>>[
    <PoseLandmarkType>[PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder],
    <PoseLandmarkType>[PoseLandmarkType.leftHip, PoseLandmarkType.rightHip],
    <PoseLandmarkType>[PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow],
    <PoseLandmarkType>[PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist],
    <PoseLandmarkType>[PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow],
    <PoseLandmarkType>[PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist],
    <PoseLandmarkType>[PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip],
    <PoseLandmarkType>[PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip],
    <PoseLandmarkType>[PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee],
    <PoseLandmarkType>[PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle],
    <PoseLandmarkType>[PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee],
    <PoseLandmarkType>[PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle],
    <PoseLandmarkType>[PoseLandmarkType.leftAnkle, PoseLandmarkType.leftHeel],
    <PoseLandmarkType>[PoseLandmarkType.rightAnkle, PoseLandmarkType.rightHeel],
    <PoseLandmarkType>[PoseLandmarkType.leftHeel, PoseLandmarkType.leftFootIndex],
    <PoseLandmarkType>[PoseLandmarkType.rightHeel, PoseLandmarkType.rightFootIndex],
  ];

  @override
  void paint(Canvas canvas, Size size) {
    _drawSkeleton(canvas);
    _drawLandmarks(canvas);
  }

  void _drawSkeleton(Canvas canvas) {
    final paint = Paint()
      ..color = const Color(0xFF16A34A)
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    for (final pair in _connections) {
      final p1 = pose[pair[0]];
      final p2 = pose[pair[1]];
      if (p1 == null || p2 == null) continue;

      final start = _toCanvasOffset(Offset(p1.x, p1.y));
      final end = _toCanvasOffset(Offset(p2.x, p2.y));
      canvas.drawLine(start, end, paint);
    }
  }

  void _drawLandmarks(Canvas canvas) {
    final high = Paint()..color = const Color(0xFF06B6D4);
    final low = Paint()..color = const Color(0xFFF59E0B);
    for (final point in pose.landmarks.values) {
      final canvasPoint = _toCanvasOffset(Offset(point.x, point.y));
      final paint = point.likelihood > 0.6 ? high : low;
      canvas.drawCircle(canvasPoint, 4.5, paint);
    }
  }

  Offset _toCanvasOffset(Offset source) {
    final sx = canvasSize.width / imageSize.width;
    final sy = canvasSize.height / imageSize.height;
    final mappedX = source.dx * sx;
    final mappedY = source.dy * sy;
    final x = mirrorX ? (canvasSize.width - mappedX) : mappedX;
    return Offset(x, mappedY);
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) {
    return oldDelegate.pose != pose ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.canvasSize != canvasSize ||
        oldDelegate.mirrorX != mirrorX;
  }
}

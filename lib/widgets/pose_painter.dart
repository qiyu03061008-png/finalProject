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
  final shadowPaint = Paint()
    ..color = Colors.black.withValues(alpha: 0.32)
    ..strokeWidth = 7
    ..strokeCap = StrokeCap.round
    ..style = PaintingStyle.stroke;

  final bonePaint = Paint()
    ..color = const Color(0xFF2DD4BF)
    ..strokeWidth = 4
    ..strokeCap = StrokeCap.round
    ..style = PaintingStyle.stroke;

  for (final pair in _connections) {
    final p1 = pose[pair[0]];
    final p2 = pose[pair[1]];
    if (p1 == null || p2 == null) continue;

    if (p1.likelihood < 0.18 || p2.likelihood < 0.18) continue;

    final start = _toCanvasOffset(Offset(p1.x, p1.y));
    final end = _toCanvasOffset(Offset(p2.x, p2.y));

    canvas.drawLine(start, end, shadowPaint);
    canvas.drawLine(start, end, bonePaint);
  }
}
//把每个关键点画成圆点
  void _drawLandmarks(Canvas canvas) {
  final outerPaint = Paint()
    ..color = Colors.white.withValues(alpha: 0.92);

  final highPaint = Paint()
    ..color = const Color(0xFF22D3EE);

  final lowPaint = Paint()
    ..color = const Color(0xFFFBBF24);

  for (final point in pose.landmarks.values) {
    if (point.likelihood < 0.18) continue;

    final canvasPoint = _toCanvasOffset(Offset(point.x, point.y));
    final paint = point.likelihood > 0.6 ? highPaint : lowPaint;

    canvas.drawCircle(canvasPoint, 6.2, outerPaint);
    canvas.drawCircle(canvasPoint, 4.2, paint);
  }
}
//把模型输出的图像坐标映射到屏幕画布坐标。
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

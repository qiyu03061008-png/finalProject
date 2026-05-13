import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/analysis_result.dart';
import '../models/pose_landmark.dart';
import '../models/user_profile.dart';
import '../services/audio_cue_service.dart';
import '../services/blazepose_detector.dart';
import '../services/pose_analyzer.dart';
import '../services/pose_depth_estimator.dart';
import '../services/tflite_pose_detector.dart';
import '../services/user_profile_repository.dart';
import '../widgets/pose_painter.dart';

class PoseDetectionScreen extends StatefulWidget {
  const PoseDetectionScreen({
    super.key,
    required this.exerciseType,
  });

  final ExerciseType exerciseType;

  /// 创建姿态检测页面对应的可变状态对象。
  @override
  State<PoseDetectionScreen> createState() => _PoseDetectionScreenState();
}

class _PoseDetectionScreenState extends State<PoseDetectionScreen> {
  final _primaryDetector = BlazePoseDetector();
  final _moveNetDetector = TFLitePoseDetector();
  final _depthEstimator = PoseDepthEstimator();
  final _analyzer = PoseAnalyzer();
  final _profileRepo = UserProfileRepository();
  final _audioCueService = AudioCueService();
  final ValueNotifier<Pose?> _poseNotifier = ValueNotifier<Pose?>(null);
  final ValueNotifier<ExerciseAnalysisResult?> _analysisNotifier =
      ValueNotifier<ExerciseAnalysisResult?>(null);

  CameraController? _primaryController;
  CameraDescription? _primaryCamera;

  bool _ready = false;
  bool _cameraPermissionDenied = false;
  bool _cameraPermissionPermanentlyDenied = false;
  bool _processingPrimary = false;
  CameraLensDirection _cameraLensDirection = CameraLensDirection.front;

  Pose? _currentPose;
  ExerciseAnalysisResult? _analysis;
  UserProfile _profile = UserProfile.defaultProfile();

  String _status = '正在初始化相机与模型...';
  int _cameraSession = 0;
  bool _voiceEnabled = true;
  bool _moveNetReady = false;
  bool _useMoveNet = false;
  int _fps = 0;
  int _framesThisSecond = 0;
  DateTime _fpsWindowStart = DateTime.now();
  double _avgInferenceMs = 0;
  int _inferenceSamples = 0;
  double _avgE2ELatencyMs = 0;
  int _e2eSamples = 0;
  int _droppedPrimaryFrames = 0;
  DateTime _lastInferenceAt = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastUiPublishAt = DateTime.fromMillisecondsSinceEpoch(0);

// 推理不要追求每帧都做，先保证相机预览不卡
  static const int _blazePoseTargetIntervalMs = 95; // 约10FPS
  static const int _moveNetTargetIntervalMs = 125;  // 约8FPS

// UI骨架刷新也要限频，避免频繁重绘压垮预览
  static const int _blazePoseUiIntervalMs = 85;
  static const int _moveNetUiIntervalMs = 110;
  DateTime _sessionStartedAt = DateTime.now();
  bool _tenMinuteReportWritten = false;
  Timer? _stabilityTimer;
  DateTime? _lastVoiceAt;
  String _lastVoiceCueKey = '';
  DateTime _lastPrimaryFrameAt = DateTime.now();
  Timer? _cameraHealthTimer;
  bool _cameraRecovering = false;
  int _noPoseFrameCount = 0;
  bool _actionRecognitionArmed = false;
  int _uprightStableFrames = 0;
  static const int _uprightRequiredStableFrames = 3;
  Pose? _lastStablePose;
  DateTime? _lastPoseSeenAt;
  static const int _keepPoseAliveMs = 220;
  Pose? _lastSmoothedMoveNetPose;
  Pose? _lastSmoothedBlazePose;
  /// 初始化页面状态，并启动相机和模型的整体初始化流程。
  @override
  void initState() {
    super.initState();
    _boot(); //启动设备和模型
  }

  /// 启动页面所需的资源初始化，包括配置、相机和后台模型。
  Future<void> _boot() async {
    try {
      _sessionStartedAt = DateTime.now();
      _startStabilityTimer();
      _profile = await _profileRepo
          .loadOrDefault()
          .timeout(const Duration(seconds: 2), onTimeout: () {
        return UserProfile.defaultProfile();
      });
      await _initCameras();
      unawaited(_initMoveNetInBackground()); //后台加载moveNet模型
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _ready = false;
        _status = '初始化失败，请重试。';
      });
    }
  }

  /// 启动稳定性统计定时器，按分钟检查是否需要落盘报告。
  void _startStabilityTimer() {
    _stabilityTimer?.cancel();
    _stabilityTimer = Timer.periodic(const Duration(minutes: 1), (_) {
      final elapsed = DateTime.now().difference(_sessionStartedAt);
      if (!_tenMinuteReportWritten && elapsed.inMinutes >= 10) {
        _tenMinuteReportWritten = true;
        unawaited(_persistStabilityReport(tenMinuteReached: true));
      }
    });
  }

  /// 在后台异步初始化 MoveNet，避免阻塞主相机流程。
  Future<void> _initMoveNetInBackground() async {

    try {
      await _moveNetDetector.initialize().timeout(const Duration(seconds: 3));
      _moveNetReady = _moveNetDetector.isReady;
      if (!mounted) return;
      setState(() {});
    } catch (_) {
      _moveNetReady = false;
    }
  }

  /// 初始化相机权限、相机列表和主相机控制器。
  Future<void> _initCameras() async {
    final sessionId = ++_cameraSession;
    setState(() {
      _ready = false;
      _status = '正在申请相机权限...';
    });

    final hasPermission = await _ensureCameraPermission(); //问用户是否能启动相机
    if (!hasPermission || !mounted || sessionId != _cameraSession) {
      return;
    }

    setState(() {
      _status = '正在读取摄像头列表...';
    });

    List<CameraDescription> deviceCameras;
    try {
      deviceCameras = await availableCameras();
    } catch (_) {
      if (!mounted || sessionId != _cameraSession) return;
      setState(() {
        _ready = false;
        _status = '无法读取当前设备的摄像头列表。';
      });
      return;
    }

    if (deviceCameras.isEmpty) {
      if (!mounted || sessionId != _cameraSession) return;
      setState(() {
        _ready = false;
        _status = '未检测到摄像头';
      });
      return;
    }

    var targetDirection = _cameraLensDirection;
    var directionCameras =
        deviceCameras.where((c) => c.lensDirection == targetDirection).toList();
    if (directionCameras.isEmpty) {
      targetDirection = deviceCameras.first.lensDirection;
      directionCameras = deviceCameras
          .where((c) => c.lensDirection == targetDirection)
          .toList();
      _cameraLensDirection = targetDirection;
    }

    _primaryCamera = directionCameras.first;

    try {
      await _primaryController?.dispose();
      if (!mounted || sessionId != _cameraSession) return;
      setState(() {
        _status = '正在初始化主摄像头...';
      });
      final primary = await _createPrimaryController(_primaryCamera!);
      if (primary == null) {
        if (!mounted || sessionId != _cameraSession) return;
        setState(() {
          _status = '相机初始化失败，请检查权限后重试。';
          _ready = false;
        });
        return;
      }

      if (!mounted || sessionId != _cameraSession) {
        await primary.dispose();
        return;
      }

      setState(() {
        _cameraPermissionDenied = false;
        _cameraPermissionPermanentlyDenied = false;
        _primaryController = primary;
        _ready = true;
        _status = '单目模式已启用（识别已启动）';
      });
      _startCameraHealthTimer();
      unawaited(_startPrimaryStream(primary, sessionId));
    } catch (_) {
      if (!mounted || sessionId != _cameraSession) return;
      setState(() {
        _status = '相机初始化失败，请检查权限后重试。';
        _ready = false;
      });
    }
  }
    //打开相机图像流
  /// 启动主相机图像流，并把每一帧交给识别逻辑处理。
  Future<void> _startPrimaryStream(
    CameraController controller,
    int sessionId,
  ) async {
    try {
      await controller.startImageStream((image) {
        _lastPrimaryFrameAt = DateTime.now();
        _onPrimaryFrame(image, sessionId);
      });
      if (!mounted || sessionId != _cameraSession) return;
      setState(() {
        _status = '单目模式已启用（识别已启动）';
      });
    } catch (_) {
      if (!mounted || sessionId != _cameraSession) return;
      setState(() {
        _status = '识别流启动失败，仅显示预览画面。';
      });
    }
  }

  /// 按候选配置依次尝试创建可用的主相机控制器。
  Future<CameraController?> _createPrimaryController(
    CameraDescription camera,
  ) async {
    final moveNetFormat = defaultTargetPlatform == TargetPlatform.android
        ? ImageFormatGroup.yuv420
        : ImageFormatGroup.bgra8888;

    final blazePoseFormat = defaultTargetPlatform == TargetPlatform.android
        ? ImageFormatGroup.nv21
        : ImageFormatGroup.bgra8888;

    final candidates = _useMoveNet
        ? <(ResolutionPreset, ImageFormatGroup?)>[
      (ResolutionPreset.low, moveNetFormat),
      (ResolutionPreset.low, null),
    ]
        : <(ResolutionPreset, ImageFormatGroup?)>[
      (ResolutionPreset.low, blazePoseFormat),
      (ResolutionPreset.low, null),
    ];
    for (final candidate in candidates) {
      CameraController? controller;
      try {
        controller = CameraController(
          camera,
          candidate.$1,
          enableAudio: false,
          imageFormatGroup: candidate.$2,
        );
        await controller.initialize().timeout(const Duration(seconds: 10));
        return controller;
      } catch (_) {
        await controller?.dispose();
      }
    }
    return null;
  }

  /// 检查并申请相机权限，同时更新页面上的权限状态提示。
  Future<bool> _ensureCameraPermission() async {
    var status = await Permission.camera.status;
    if (!status.isGranted) {
      status = await Permission.camera.request();
    }
    if (status.isGranted) {
      if (!mounted) return true;
      setState(() {
        _cameraPermissionDenied = false;
        _cameraPermissionPermanentlyDenied = false;
      });
      return true;
    }
    if (!mounted) return false;
    final permanentlyDenied = status.isPermanentlyDenied || status.isRestricted;
    setState(() {
      _cameraPermissionDenied = true;
      _cameraPermissionPermanentlyDenied = permanentlyDenied;
      _ready = false;
      _status = permanentlyDenied ? '相机权限已被禁用，请前往设置开启。' : '启动检测需要相机权限。';
    });
    return false;
  }

  /// 处理主相机的一帧图像，完成检测、补深度、分析和界面更新。
  /// 核心
  Future<void> _onPrimaryFrame(CameraImage image, int sessionId) async {
    final camera = _primaryCamera;
    if (_processingPrimary) {
      _droppedPrimaryFrames += 1;
      return;
    }
    if (camera == null || sessionId != _cameraSession) {
      return;
    }
    final now = DateTime.now();
    final targetIntervalMs =
    _useMoveNet ? _moveNetTargetIntervalMs : _blazePoseTargetIntervalMs;

    if (now.difference(_lastInferenceAt).inMilliseconds < targetIntervalMs) {
      return;
    }
    _lastInferenceAt = now;

    _processingPrimary = true;
    final capturedAt = DateTime.now();
    final stopwatch = Stopwatch()..start();

    try {
      final pose2d = _useMoveNet
          ? await _moveNetDetector.detectPoseFromCamera(image, camera)
          : await _primaryDetector.detectFromCameraImage(
              image: image,
              camera: camera,
              captureTime: capturedAt,
            );
      if (sessionId != _cameraSession) return;

      if (pose2d == null) {
        _noPoseFrameCount += 1;
        final canHoldLastPose = _lastStablePose != null &&
            _lastPoseSeenAt != null &&
            DateTime.now().difference(_lastPoseSeenAt!).inMilliseconds <=
                _keepPoseAliveMs;
        if (_noPoseFrameCount >= 1 && !canHoldLastPose && mounted) {
          _currentPose = null;
          _analysis = null;
          _actionRecognitionArmed = false;
          _uprightStableFrames = 0;
          _lastSmoothedMoveNetPose = null;
          _lastSmoothedBlazePose = null;
          _publishLiveState(pose: null, analysis: null);
        }
        return;
      }
      _noPoseFrameCount = 0;

      final filteredPose2d = _useMoveNet
          ? _smoothMoveNetPose(pose2d)
          : _smoothBlazePose(pose2d);

      final displayPose = _depthEstimator.estimateMonocular3D(
        pose2d: filteredPose2d,
        profile: _profile, // 用用户的身高体重来校准
      );

      final fused = displayPose;

// 先确认用户站好了，才开始分析动作
      if (!_actionRecognitionArmed) {
        final uprightReady = _isUprightReadyPose(fused);
        _uprightStableFrames = uprightReady ? _uprightStableFrames + 1 : 0;
        if (_uprightStableFrames < _uprightRequiredStableFrames) {
          if (!mounted || sessionId != _cameraSession) return;
          _currentPose = null;
          _analysis = null;
          _publishLiveState(pose: null, analysis: null);
          return;
        }
        _actionRecognitionArmed = true;
        _analyzer.reset(widget.exerciseType);
      }

      final analysis = _hasReliableAnalysisPose(fused)
          ? _analyzer.analyze(
              pose: fused,
              exerciseType: widget.exerciseType,
              profile: _profile,
            )
          : null;
      _recordPerformance(stopwatch.elapsedMilliseconds.toDouble());
      if (analysis != null) {
        _maybeSpeakFeedback(analysis);
      }
      _recordE2ELatency(
        DateTime.now().difference(capturedAt).inMilliseconds.toDouble(),
      );

      if (!mounted || sessionId != _cameraSession) return;
      _lastStablePose = displayPose;
      _lastPoseSeenAt = DateTime.now();
      _currentPose = displayPose;
      _analysis = analysis;
      _publishLiveState(pose: displayPose, analysis: analysis);
    } catch (_) {
      // Ignore single-frame failures to keep stream alive.
    } finally {
      _processingPrimary = false;
    }
  }

  /// 判断用户是否已经以较稳定的直立姿态进入取景区域。
  bool _isUprightReadyPose(Pose pose) {
    PoseLandmark? point(PoseLandmarkType type) => pose[type];
    bool reliable(PoseLandmarkType type) =>
        point(type)?.likelihood != null && point(type)!.likelihood >= 0.55;

    if (!reliable(PoseLandmarkType.leftShoulder) ||
        !reliable(PoseLandmarkType.rightShoulder) ||
        !reliable(PoseLandmarkType.leftHip) ||
        !reliable(PoseLandmarkType.rightHip)) {
      return false;
    }

    final leftShoulder = point(PoseLandmarkType.leftShoulder)!;
    final rightShoulder = point(PoseLandmarkType.rightShoulder)!;
    final leftHip = point(PoseLandmarkType.leftHip)!;
    final rightHip = point(PoseLandmarkType.rightHip)!;

    final shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    final hipY = (leftHip.y + rightHip.y) / 2;
    final torsoHeight = (hipY - shoulderY).abs();
    if (torsoHeight < 28) return false;

    final shoulderTilt = (leftShoulder.y - rightShoulder.y).abs();
    final hipTilt = (leftHip.y - rightHip.y).abs();
    if (shoulderTilt > torsoHeight * 0.45) return false;
    if (hipTilt > torsoHeight * 0.45) return false;

    final kneeReliable = reliable(PoseLandmarkType.leftKnee) ||
        reliable(PoseLandmarkType.rightKnee);
    final ankleReliable = reliable(PoseLandmarkType.leftAnkle) ||
        reliable(PoseLandmarkType.rightAnkle);

    if (!(shoulderY < hipY)) return false;

    if (kneeReliable && ankleReliable) {
      final kneeY = ((point(PoseLandmarkType.leftKnee)?.y ?? 0.0) +
              (point(PoseLandmarkType.rightKnee)?.y ?? 0.0)) /
          2;
      final ankleY = ((point(PoseLandmarkType.leftAnkle)?.y ?? 0.0) +
              (point(PoseLandmarkType.rightAnkle)?.y ?? 0.0)) /
          2;
      if (kneeY <= hipY) return false;
      if (ankleY <= kneeY) return false;
    } else if (kneeReliable) {
      final kneeY = ((point(PoseLandmarkType.leftKnee)?.y ?? 0.0) +
              (point(PoseLandmarkType.rightKnee)?.y ?? 0.0)) /
          2;
      if (kneeY <= hipY) return false;
    } else if (ankleReliable) {
      final ankleY = ((point(PoseLandmarkType.leftAnkle)?.y ?? 0.0) +
              (point(PoseLandmarkType.rightAnkle)?.y ?? 0.0)) /
          2;
      if (ankleY <= hipY) return false;
    }
    return true;
  }

  /// 同步当前姿态和分析结果，并顺手统计实时帧率。
  void _publishLiveState({
    required Pose? pose,
    required ExerciseAnalysisResult? analysis,
  }) {
    final now = DateTime.now();

    final uiIntervalMs =
    _useMoveNet ? _moveNetUiIntervalMs : _blazePoseUiIntervalMs;

    if (pose != null &&
        now.difference(_lastUiPublishAt).inMilliseconds < uiIntervalMs) {
      _framesThisSecond += 1;

      if (now.difference(_fpsWindowStart).inMilliseconds >= 1000) {
        _fps = _framesThisSecond;
        _framesThisSecond = 0;
        _fpsWindowStart = now;
      }
      return;
    }

    if (pose != null) {
      _lastUiPublishAt = now;
    }

    _poseNotifier.value = pose;
    _analysisNotifier.value = analysis;

    _framesThisSecond += 1;
    if (now.difference(_fpsWindowStart).inMilliseconds >= 1000) {
      _fps = _framesThisSecond;
      _framesThisSecond = 0;
      _fpsWindowStart = now;
    }
  }

  /// 记录单次推理耗时，并更新平均推理时间。
  void _recordPerformance(double inferenceMs) {
    _inferenceSamples += 1;
    _avgInferenceMs += (inferenceMs - _avgInferenceMs) / _inferenceSamples;
  }

  /// 记录端到端延迟，并更新平均端到端耗时。
  void _recordE2ELatency(double e2eMs) {
    _e2eSamples += 1;
    _avgE2ELatencyMs += (e2eMs - _avgE2ELatencyMs) / _e2eSamples;
  }

  /// 定时检查相机是否长时间无新帧，用于自动恢复卡死情况。
  void _startCameraHealthTimer() {
    _cameraHealthTimer?.cancel();
    _cameraHealthTimer = Timer.periodic(const Duration(seconds: 3), (_) {
      if (_cameraRecovering || !_ready || _primaryController == null) {
        return;
      }
      final silenceMs =
          DateTime.now().difference(_lastPrimaryFrameAt).inMilliseconds;
      if (silenceMs < 3000) return;
      unawaited(_attemptCameraRecovery());
    });
  }

  /// 在检测到相机异常时尝试自动重建相机链路。
  Future<void> _attemptCameraRecovery() async {
    if (_cameraRecovering) return;
    _cameraRecovering = true;
    try {
      if (!mounted) return;
      setState(() {
        _status = '检测到画面卡住，正在自动恢复...';
        _ready = false;
      });
      _publishLiveState(pose: null, analysis: null);
      if (_primaryController != null) {
        await _primaryController!.dispose();
      }
      _primaryController = null;
      await _initCameras();
    } finally {
      _cameraRecovering = false;
    }
  }

  /// 对 MoveNet 输出做时序平滑，减少关键点抖动和跳点。
  Pose _smoothMoveNetPose(Pose pose) {
    const lowConfidenceHold = 0.34;
    const stillAlpha = 0.16;   // 静止时压抖
    const moveAlpha = 0.58;    // 运动时快速跟随
    const fastAlpha = 0.78;    // 大动作时更快跟随
    const maxJumpRatio = 0.85;

    final previous = _lastSmoothedMoveNetPose;
    if (previous == null) {
      _lastSmoothedMoveNetPose = pose;
      return pose;
    }

    final torsoScale = _estimateTorsoScale(pose, previous);
    final stillThreshold = torsoScale * 0.018;
    final fastThreshold = torsoScale * 0.12;
    final maxJump = torsoScale * maxJumpRatio;

    final smoothed = <PoseLandmarkType, PoseLandmark>{};

    for (final type in PoseLandmarkType.values) {
      final previousPoint = previous.landmarks[type];
      final currentPoint = pose.landmarks[type];

      if (currentPoint == null) {
        if (previousPoint != null &&
            previousPoint.likelihood >= lowConfidenceHold) {
          smoothed[type] = previousPoint.copyWith(
            likelihood: previousPoint.likelihood * 0.92,
          );
        }
        continue;
      }

      if (previousPoint == null) {
        smoothed[type] = currentPoint;
        continue;
      }

      final dx = currentPoint.x - previousPoint.x;
      final dy = currentPoint.y - previousPoint.y;
      final jump = math.sqrt(dx * dx + dy * dy);

      if (currentPoint.likelihood < lowConfidenceHold &&
          previousPoint.likelihood >= lowConfidenceHold) {
        smoothed[type] = previousPoint.copyWith(
          likelihood: math.max(
            previousPoint.likelihood * 0.90,
            currentPoint.likelihood,
          ),
        );
        continue;
      }

      if (jump > maxJump &&
          previousPoint.likelihood >= currentPoint.likelihood * 0.9) {
        smoothed[type] = previousPoint.copyWith(
          likelihood: previousPoint.likelihood * 0.88,
        );
        continue;
      }

      final alpha = jump < stillThreshold
          ? stillAlpha
          : jump > fastThreshold
          ? fastAlpha
          : moveAlpha;

      smoothed[type] = PoseLandmark(
        x: previousPoint.x * (1 - alpha) + currentPoint.x * alpha,
        y: previousPoint.y * (1 - alpha) + currentPoint.y * alpha,
        z: previousPoint.z * (1 - alpha) + currentPoint.z * alpha,
        likelihood: currentPoint.likelihood,
      );
    }

    final nextPose = pose.copyWith(landmarks: smoothed);
    _lastSmoothedMoveNetPose = nextPose;
    return nextPose;
  }

  Pose _smoothBlazePose(Pose pose) {
    const lowConfidenceHold = 0.42;
    const stillAlpha = 0.22;
    const moveAlpha = 0.62;
    const fastAlpha = 0.82;
    const maxJumpRatio = 0.95;

    final previous = _lastSmoothedBlazePose;
    if (previous == null) {
      _lastSmoothedBlazePose = pose;
      return pose;
    }

    final torsoScale = _estimateTorsoScale(pose, previous);
    final stillThreshold = torsoScale * 0.018;
    final fastThreshold = torsoScale * 0.12;
    final maxJump = torsoScale * maxJumpRatio;

    final smoothed = <PoseLandmarkType, PoseLandmark>{};

    for (final type in PoseLandmarkType.values) {
      final previousPoint = previous.landmarks[type];
      final currentPoint = pose.landmarks[type];

      if (currentPoint == null) {
        if (previousPoint != null &&
            previousPoint.likelihood >= lowConfidenceHold) {
          smoothed[type] = previousPoint.copyWith(
            likelihood: previousPoint.likelihood * 0.94,
          );
        }
        continue;
      }

      if (previousPoint == null) {
        smoothed[type] = currentPoint;
        continue;
      }

      final dx = currentPoint.x - previousPoint.x;
      final dy = currentPoint.y - previousPoint.y;
      final jump = math.sqrt(dx * dx + dy * dy);

      if (currentPoint.likelihood < lowConfidenceHold &&
          previousPoint.likelihood >= lowConfidenceHold) {
        smoothed[type] = previousPoint.copyWith(
          likelihood: math.max(
            previousPoint.likelihood * 0.92,
            currentPoint.likelihood,
          ),
        );
        continue;
      }

      if (jump > maxJump &&
          previousPoint.likelihood >= currentPoint.likelihood * 0.9) {
        smoothed[type] = previousPoint.copyWith(
          likelihood: previousPoint.likelihood * 0.90,
        );
        continue;
      }

      final alpha = jump < stillThreshold
          ? stillAlpha
          : jump > fastThreshold
          ? fastAlpha
          : moveAlpha;

      smoothed[type] = PoseLandmark(
        x: previousPoint.x * (1 - alpha) + currentPoint.x * alpha,
        y: previousPoint.y * (1 - alpha) + currentPoint.y * alpha,
        z: previousPoint.z * (1 - alpha) + currentPoint.z * alpha,
        likelihood: currentPoint.likelihood,
      );
    }

    final nextPose = pose.copyWith(landmarks: smoothed);
    _lastSmoothedBlazePose = nextPose;
    return nextPose;
  }

  /// 估计当前人体躯干尺度，用来限制关键点允许跳动的幅度。
  double _estimateTorsoScale(Pose current, Pose previous) {
    double fromPose(Pose pose) {
      final leftShoulder = pose[PoseLandmarkType.leftShoulder];
      final rightShoulder = pose[PoseLandmarkType.rightShoulder];
      final leftHip = pose[PoseLandmarkType.leftHip];
      final rightHip = pose[PoseLandmarkType.rightHip];
      if (leftShoulder != null &&
          rightShoulder != null &&
          leftHip != null &&
          rightHip != null) {
        final shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
        final hipMidY = (leftHip.y + rightHip.y) / 2;
        return (hipMidY - shoulderMidY).abs();
      }
      return 0;
    }

    final currentScale = fromPose(current);
    final previousScale = fromPose(previous);
    return math.max(36, math.max(currentScale, previousScale));
  }

  /// 判断当前姿态点是否足够可靠，值得继续进入动作分析阶段。
  bool _hasReliableAnalysisPose(Pose pose) {
    final torso = <PoseLandmark?>[
      pose[PoseLandmarkType.leftShoulder],
      pose[PoseLandmarkType.rightShoulder],
      pose[PoseLandmarkType.leftHip],
      pose[PoseLandmarkType.rightHip],
    ];
    final isMoveNet = pose.source.startsWith('movenet');
    final torsoThreshold = isMoveNet ? 0.32 : 0.45;
    final landmarkThreshold = isMoveNet ? 0.24 : 0.35;
    final minReliableLandmarks = isMoveNet ? 6 : 8;
    final torsoReliable =
        torso.where((p) => p != null && p.likelihood >= torsoThreshold).length;
    if (torsoReliable < 3) return false;

    final reliableCount = pose.landmarks.values
        .where((landmark) => landmark.likelihood >= landmarkThreshold)
        .length;
    return reliableCount >= minReliableLandmarks;
  }

  /// 按节流规则播放语音反馈，避免过于频繁地打断用户。
  Future<void> _maybeSpeakFeedback(ExerciseAnalysisResult analysis) async {
    if (!_voiceEnabled) return;
    if (analysis.issues.isEmpty && !analysis.repJustCountedClean) {
      return;
    }
    final now = DateTime.now();
    final cueKey = analysis.issues.isNotEmpty
        ? analysis.issues.first.type.name
        : '${widget.exerciseType.name}_clean_rep';

    final lastAt = _lastVoiceAt;
    if (lastAt != null && now.difference(lastAt).inSeconds < 3) return;
    if (cueKey == _lastVoiceCueKey &&
        lastAt != null &&
        now.difference(lastAt).inSeconds < 6) {
      return;
    }
    _lastVoiceAt = now;
    _lastVoiceCueKey = cueKey;
    await _audioCueService.playForAnalysis(
      analysis: analysis,
      exerciseType: widget.exerciseType,
    );
  }

  /// 切换前后摄像头，并重置当前识别会话的瞬时状态。
  Future<void> _toggleCameraLensDirection() async {
    setState(() {
      _ready = false;
      _status = '正在切换摄像头...';
      _currentPose = null;
      _lastStablePose = null;
      _lastSmoothedMoveNetPose = null;
      _lastSmoothedBlazePose = null;
      _lastPoseSeenAt = null;
      _analysis = null;
      _actionRecognitionArmed = false;
      _uprightStableFrames = 0;
    });

    _publishLiveState(pose: null, analysis: null);
    _cameraSession += 1;

    if (_primaryController != null) {
      await _primaryController!.dispose();
    }
    _primaryController = null;

    _cameraLensDirection = _cameraLensDirection == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    await _initCameras();
  }

  /// 释放页面持有的控制器、定时器和识别资源。
  @override
  void dispose() {
    _stabilityTimer?.cancel();
    _cameraHealthTimer?.cancel();
    unawaited(
      _persistStabilityReport(tenMinuteReached: _tenMinuteReportWritten),
    );
    _primaryController?.dispose();
    _moveNetDetector.dispose();
    _audioCueService.stop();
    _audioCueService.dispose();
    _poseNotifier.dispose();
    _analysisNotifier.dispose();
    _primaryDetector.close();
    super.dispose();
  }

  /// 把当前会话的稳定性统计写入本地日志文件。
  Future<void> _persistStabilityReport({required bool tenMinuteReached}) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final reportFile = File('${dir.path}/stability_reports.jsonl');
      final elapsed = DateTime.now().difference(_sessionStartedAt);
      final report = <String, dynamic>{
        'timestamp': DateTime.now().toIso8601String(),
        'exercise': widget.exerciseType.name,
        'session_duration_sec': elapsed.inSeconds,
        'ten_minute_reached': tenMinuteReached,
        'avg_fps': _fps,
        'avg_inference_ms': _avgInferenceMs,
        'avg_end_to_end_ms': _avgE2ELatencyMs,
        'dropped_primary_frames': _droppedPrimaryFrames,
        'depth_mode': 'monocular',
      };
      await reportFile.writeAsString(
        '${jsonEncode(report)}\n',
        mode: FileMode.append,
      );
    } catch (_) {
      // Keep stability logging best-effort and non-blocking.
    }
  }

  /// 构建姿态检测页面，包括相机预览、信息卡片和反馈区域。
  @override
  Widget build(BuildContext context) {
    final controller = _primaryController;
    return Scaffold(
      backgroundColor: Colors.transparent,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        foregroundColor: Colors.white,
        title: Text(widget.exerciseType.label),
        actions: <Widget>[
          IconButton(
            tooltip: !_moveNetReady
                ? 'MoveNet 模型不可用，请确认 assets/models/movenet_lightning.tflite 已打包'
                : _useMoveNet
                    ? '切换到 BlazePose'
                    : '切换到 MoveNet',
            onPressed: !_moveNetReady
                ? null
                : () async {
                    setState(() {
                      _useMoveNet = !_useMoveNet;
                      _ready = false;
                      _status = _useMoveNet
                          ? '正在切换到 MoveNet...'
                          : '正在切换到 BlazePose...';
                      _currentPose = null;
                      _lastStablePose = null;
                      _lastSmoothedMoveNetPose = null;
                      _lastSmoothedBlazePose = null;
                      _lastPoseSeenAt = null;
                      _analysis = null;
                      _actionRecognitionArmed = false;
                      _uprightStableFrames = 0;
                    });
                    _lastInferenceAt = DateTime.fromMillisecondsSinceEpoch(0);
                    _lastUiPublishAt = DateTime.fromMillisecondsSinceEpoch(0);
                    _publishLiveState(pose: null, analysis: null);
                    _cameraSession += 1;
                    await _primaryController?.dispose();
                    _primaryController = null;
                    if (mounted) {
                      await _initCameras();
                    }
                  },
            icon: Icon(
              _useMoveNet
                  ? Icons.directions_run
                  : Icons.directions_run_outlined,
            ),
          ),
          IconButton(
            tooltip: _voiceEnabled ? '关闭语音反馈' : '开启语音反馈',
            onPressed: () {
              setState(() {
                _voiceEnabled = !_voiceEnabled;
              });
              if (!_voiceEnabled) {
                _audioCueService.stop();
              }
            },
            icon: Icon(_voiceEnabled ? Icons.volume_up : Icons.volume_off),
          ),
          IconButton(
            tooltip: _cameraLensDirection == CameraLensDirection.front
                ? '切换到后置摄像头'
                : '切换到前置摄像头',
            onPressed: _toggleCameraLensDirection,
            icon: const Icon(Icons.cameraswitch),
          ),
          IconButton(
            tooltip: '重置计数',
            onPressed: () {
              _analyzer.reset(widget.exerciseType);
              _publishLiveState(pose: _currentPose, analysis: null);
            },
            icon: const Icon(Icons.restart_alt),
          ),
        ],
      ),
      body: !_ready || controller == null
          ? _buildLoadingOrErrorState()
          : Stack(
              fit: StackFit.expand,
              children: <Widget>[
                _buildAdaptiveCameraPreview(controller),
                Positioned(
                  top: 10,
                  left: 10,
                  right: 10,
                  child: ValueListenableBuilder<ExerciseAnalysisResult?>(
                    valueListenable: _analysisNotifier,
                    builder: (context, analysis, _) => _buildInfoCard(analysis),
                  ),
                ),
                Positioned(
                  bottom: 16,
                  left: 16,
                  right: 16,
                  child: ValueListenableBuilder<ExerciseAnalysisResult?>(
                    valueListenable: _analysisNotifier,
                    builder: (context, analysis, _) =>
                        _buildFeedbackCard(analysis),
                  ),
                ),
              ],
            ),
    );
  }

  /// 构建加载中或错误状态下的占位界面。
  Widget _buildLoadingOrErrorState() {
    final showActions = _cameraPermissionDenied || _status.contains('失败');
    final buttonLabel = _cameraPermissionPermanentlyDenied
        ? '打开应用设置'
        : _cameraPermissionDenied
            ? '授予相机权限'
            : '重试';
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Container(
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            color: const Color(0xFF0F172A).withValues(alpha: 0.82),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              Container(
                width: 58,
                height: 58,
                decoration: BoxDecoration(
                  color: const Color(0xFF14B8A6).withValues(alpha: 0.18),
                  shape: BoxShape.circle,
                ),
                child: const Icon(
                  Icons.videocam_rounded,
                  color: Color(0xFF5EEAD4),
                  size: 30,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                _status,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  height: 1.5,
                ),
              ),
              if (showActions) ...<Widget>[
                const SizedBox(height: 16),
                FilledButton(
                  onPressed: () async {
                    if (_cameraPermissionPermanentlyDenied) {
                      await openAppSettings();
                      await _initCameras();
                      return;
                    }
                    await _initCameras();
                  },
                  child: Text(buttonLabel),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  /// 构建自适应尺寸的相机预览，并叠加姿态骨架绘制层。
  Widget _buildAdaptiveCameraPreview(CameraController controller) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final previewSize = controller.value.previewSize;
        if (previewSize == null) {
          return const Center(child: CircularProgressIndicator());
        }

        final rawPreviewSize = Size(previewSize.height, previewSize.width);

        return ClipRect(
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: rawPreviewSize.width,
              height: rawPreviewSize.height,
              child: Stack(
                fit: StackFit.expand,
                children: <Widget>[
                  CameraPreview(controller),
                  ValueListenableBuilder<Pose?>(
                    valueListenable: _poseNotifier,
                    builder: (context, pose, _) {
                      if (pose == null) {
                        return const SizedBox.shrink();
                      }
                      return RepaintBoundary(
                        child: CustomPaint(
                          painter: PosePainter(
                            pose: pose,
                            imageSize: rawPreviewSize,
                            canvasSize: rawPreviewSize,
                            mirrorX: _cameraLensDirection ==
                                CameraLensDirection.front,
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  /// 构建顶部信息卡，展示动作、分数、延迟和帧率等信息。
  Widget _buildInfoCard(ExerciseAnalysisResult? analysis) {
    final countLabel =
        widget.exerciseType == ExerciseType.plank ? '保持(秒)' : '次数';
    final countValue = analysis?.count ?? 0;
    final score = analysis?.score ?? 0;
    final depthLabel = analysis?.depthModeLabel ?? '单目 3D';
    final detectorLabel = _useMoveNet ? 'MoveNet' : 'BlazePose';

    final fpsLabel = _fps <= 0 ? '--' : '$_fps';
    final latencyLabel =
        _inferenceSamples == 0 ? '--' : _avgInferenceMs.toStringAsFixed(0);
    final e2eLabel =
        _e2eSamples == 0 ? '--' : _avgE2ELatencyMs.toStringAsFixed(0);
    final dropLabel = '$_droppedPrimaryFrames';
    return Align(
      alignment: Alignment.topCenter,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 360),
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 5),
        decoration: BoxDecoration(
          color: Colors.transparent,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
        ),
        child: Wrap(
          spacing: 4,
          runSpacing: 4,
          alignment: WrapAlignment.center,
          children: <Widget>[
            _metricItem('动作', widget.exerciseType.label),
            _metricItem(countLabel, '$countValue'),
            _metricItem('得分', '$score'),
            _metricItem('深度', depthLabel),
            _metricItem('检测器', detectorLabel),
            _metricItem('帧率', fpsLabel),
            _metricItem('延迟', '${latencyLabel}ms'),
            _metricItem('端到端', '${e2eLabel}ms'),
            _metricItem('丢帧', dropLabel),
            _metricItem('语音', _voiceEnabled ? '开' : '关'),
          ],
        ),
      ),
    );
  }

  /// 构建单个指标展示块。
  Widget _metricItem(String label, String value) {
    return Container(
      width: 64,
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(fontSize: 8, color: Colors.white60),
          ),
          const SizedBox(height: 1),
          Text(
            value,
            style: const TextStyle(
              fontSize: 11,
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  /// 构建底部反馈卡，展示当前提示语和纠正建议。
  Widget _buildFeedbackCard(ExerciseAnalysisResult? analysis) {
    final issue = analysis != null && analysis.issues.isNotEmpty
        ? analysis.issues.first
        : null;
    final feedbackText = !_actionRecognitionArmed
        ? '请先完整站立进入镜头，再开始动作识别'
        : analysis?.feedback ?? '请先完成一个标准动作后再开始提示';

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.transparent,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.18)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.16),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text(
                  issue == null ? '实时反馈' : '动作纠正',
                  style: TextStyle(
                    color: issue == null
                        ? const Color(0xFF5EEAD4)
                        : const Color(0xFFFDA4AF),
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            feedbackText,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 17,
              fontWeight: FontWeight.w700,
              height: 1.45,
            ),
          ),
          if (issue != null) ...<Widget>[
            const SizedBox(height: 10),
            Text(
              '建议：${issue.suggestion}',
              style: const TextStyle(color: Color(0xFF93C5FD), fontSize: 13),
            ),
          ],
        ],
      ),
    );
  }
}

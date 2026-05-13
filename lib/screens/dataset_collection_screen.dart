import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/analysis_result.dart';
import '../models/pose_landmark.dart';
import '../models/user_profile.dart';
import '../services/blazepose_detector.dart';
import '../services/dataset_collection_service.dart';
import '../services/user_profile_repository.dart';
import '../widgets/pose_painter.dart';
import 'annotation_correction_screen.dart';

class DatasetCollectionScreen extends StatefulWidget {
  const DatasetCollectionScreen({super.key});

  @override
  State<DatasetCollectionScreen> createState() =>
      _DatasetCollectionScreenState();
}

class _DatasetCollectionScreenState extends State<DatasetCollectionScreen> {
  final _primaryDetector = BlazePoseDetector(strictHumanValidation: false);
  final _datasetService = DatasetCollectionService();
  final _profileRepo = UserProfileRepository();
  final _subjectController = TextEditingController(text: '受试者01');
  final _annotatorController = TextEditingController(text: '标注员A');

  CameraController? _primaryController;
  CameraDescription? _primaryCamera;
  Pose? _pose;
  bool _ready = false;
  bool _cameraPermissionDenied = false;
  bool _cameraPermissionPermanentlyDenied = false;
  bool _busy = false;
  bool _saving = false;
  CameraLensDirection _cameraLensDirection = CameraLensDirection.front;
  String _status = '初始化中...';
  String? _lastSavedPath;
  Map<PoseLandmarkType, PoseLandmark>? _correctedLandmarks;
  Pose? _manualPoseSnapshot;

  ExerciseType _exerciseType = ExerciseType.squat;
  String _qualityTag = 'standard';
  String _viewTag = 'front';
  bool _manualChecked = false;
  UserProfile _profile = UserProfile.defaultProfile();
  DatasetProgress _progress = DatasetProgress.empty();
  int _poseStableFrames = 0;
  int _noPoseFrames = 0;
  static const int _poseRequiredStableFrames = 3;

  @override
  void initState() {
    super.initState();
    _initAll();
  }

  Future<void> _initAll() async {
    try {
      _profile = await _profileRepo
          .loadOrDefault()
          .timeout(const Duration(seconds: 2), onTimeout: () {
        return UserProfile.defaultProfile();
      });
      _progress = await _datasetService
          .loadProgress()
          .timeout(const Duration(seconds: 2), onTimeout: () {
        return DatasetProgress.empty();
      });
      await _initCamera();
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _ready = false;
        _status = '初始化失败，请重试。';
      });
    }
  }

  Future<void> _initCamera() async {
    setState(() {
      _ready = false;
      _status = '正在申请相机权限...';
    });

    final hasPermission = await _ensureCameraPermission();
    if (!hasPermission) {
      return;
    }
    if (!mounted) return;
    setState(() => _status = '正在读取摄像头列表...');

    List<CameraDescription> deviceCameras;
    try {
      deviceCameras = await availableCameras();
    } catch (_) {
      if (!mounted) return;
      setState(() => _status = '无法读取当前设备的摄像头列表');
      return;
    }
    if (deviceCameras.isEmpty) {
      if (!mounted) return;
      setState(() => _status = '没有可用摄像头');
      return;
    }

    var targetDirection = _cameraLensDirection;
    var directionCameras = deviceCameras
        .where((c) => c.lensDirection == targetDirection)
        .toList();

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
      if (!mounted) return;
      setState(() => _status = '正在初始化主摄像头...');
      final primary = await _createPrimaryController(_primaryCamera!);
      if (primary == null) {
        if (!mounted) return;
        setState(() {
          _ready = false;
          _status = '相机初始化失败，请检查权限后重试。';
        });
        return;
      }
      if (!mounted) {
        await primary.dispose();
        return;
      }
      setState(() {
        _cameraPermissionDenied = false;
        _cameraPermissionPermanentlyDenied = false;
        _primaryController = primary;
        _ready = true;
        _status = '摄像头预览已就绪，正在启动采集流...';
      });
      unawaited(_startPrimaryStream(primary));
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _ready = false;
        _status = '相机初始化失败，请检查权限后重试。';
      });
    }
  }

  Future<void> _toggleCameraLensDirection() async {
    setState(() {
      _ready = false;
      _status = '正在切换摄像头...';
      _pose = null;
    });

    await _primaryController?.dispose();
    _primaryController = null;

    _cameraLensDirection = _cameraLensDirection == CameraLensDirection.back
        ? CameraLensDirection.front
        : CameraLensDirection.back;

    await _initCamera();
  }

  Future<void> _startPrimaryStream(CameraController controller) async {
    try {
      await controller.startImageStream((image) {
        _onPrimaryFrame(image);
      });
      if (!mounted) return;
      setState(() {
        _status = '单目采集已就绪（采集流已启动）';
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _status = '采集流启动失败，仅显示预览画面。';
      });
    }
  }

  Future<CameraController?> _createPrimaryController(
    CameraDescription camera,
  ) async {
    final candidates = <(ResolutionPreset, ImageFormatGroup?)>[
      (ResolutionPreset.low, null),
      (ResolutionPreset.medium, null),
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
      _status = permanentlyDenied
          ? '相机权限已被禁用，请前往设置开启。'
          : '数据采集需要相机权限。';
    });
    return false;
  }

  //每一帧图像处理流程（核心）
  Future<void> _onPrimaryFrame(CameraImage image) async {
    final camera = _primaryCamera;
    if (camera == null) return;
    if (_busy) return;
    _busy = true;
    final capturedAt = DateTime.now();
    try {
      final pose = await _primaryDetector.detectFromCameraImage(
        image: image,
        camera: camera,
        captureTime: capturedAt,
      );
      if (!mounted) return;
      if (pose != null) {
        _noPoseFrames = 0;
        _poseStableFrames += 1;
        if (_poseStableFrames >= _poseRequiredStableFrames) {
          setState(() {
            _pose = pose;
            if (_manualPoseSnapshot != null &&
                pose.timestamp != _manualPoseSnapshot!.timestamp) {
              _correctedLandmarks = null;
              _manualPoseSnapshot = null;
            }
          });
        }
      } else {
        _poseStableFrames = 0;
        _noPoseFrames += 1;
        if (_noPoseFrames >= _poseRequiredStableFrames) {
          setState(() {
            _pose = null;
          });
        }
      }
    } catch (_) {
      // Ignore single-frame inference failures.
    } finally {
      _busy = false;
    }
  }

  @override
  void dispose() {
    _primaryController?.dispose();
    _primaryDetector.close();
    _subjectController.dispose();
    _annotatorController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _primaryController;
    return Scaffold(
      appBar: AppBar(
        title: const Text('数据集采集'),
        actions: <Widget>[
          IconButton(
            tooltip: _cameraLensDirection == CameraLensDirection.front
                ? '切换到后置摄像头'
                : '切换到前置摄像头',
            onPressed: _toggleCameraLensDirection,
            icon: const Icon(Icons.cameraswitch),
          ),
        ],
      ),
      body: !_ready || controller == null
          ? _buildLoadingOrErrorState()
          : Column(
              children: <Widget>[
                Expanded(child: _buildAdaptiveCameraPreview(controller)),
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    children: <Widget>[
                      _buildProgressCard(),
                      const SizedBox(height: 10),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: <Widget>[
                          _exerciseDropdown(),
                          _tagDropdown(
                            value: _qualityTag,
                            label: '质量',
                            items: const <String>['standard', 'error'],
                            onChanged: (v) => setState(() => _qualityTag = v),
                          ),
                          _tagDropdown(
                            value: _viewTag,
                            label: '视角',
                            items: const <String>['front', 'side', 'oblique'],
                            onChanged: (v) => setState(() => _viewTag = v),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: TextField(
                              controller: _subjectController,
                              decoration: const InputDecoration(
                                isDense: true,
                                border: OutlineInputBorder(),
                                labelText: '受试者标签',
                                hintText: '受试者01',
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: TextField(
                              controller: _annotatorController,
                              decoration: const InputDecoration(
                                isDense: true,
                                border: OutlineInputBorder(),
                                labelText: '标注员',
                                hintText: '标注员A',
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      CheckboxListTile(
                        value: _manualChecked,
                        onChanged: (v) =>
                            setState(() => _manualChecked = v ?? false),
                        contentPadding: EdgeInsets.zero,
                        title: const Text('已人工复核'),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed: _openLeftCorrection,
                              icon: const Icon(Icons.edit_note_outlined),
                              label: const Text('编辑关键点'),
                            ),
                          ),
                        ],
                      ),
                      if (_correctedLandmarks != null) ...<Widget>[
                        const SizedBox(height: 6),
                        const Text(
                          '已暂存人工修正结果',
                          style:
                              TextStyle(fontSize: 12, color: Colors.black54),
                        ),
                      ],
                      const SizedBox(height: 8),
                      Row(
                        children: <Widget>[
                          Expanded(
                            child: FilledButton.icon(
                              onPressed: _saving ? null : _saveSample,
                              icon: _saving
                                  ? const SizedBox(
                                      width: 16,
                                      height: 16,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                      ),
                                    )
                                  : const Icon(Icons.add_task_outlined),
                              label:
                                  Text(_saving ? '保存中...' : '保存样本'),
                            ),
                          ),
                          const SizedBox(width: 8),
                          OutlinedButton.icon(
                            onPressed: _exportGuideline,
                            icon: const Icon(Icons.rule_folder_outlined),
                            label: const Text('导出标注规范'),
                          ),
                        ],
                      ),
                      if (_lastSavedPath != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 8),
                          child: Text(
                            '已保存到：$_lastSavedPath',
                            style: const TextStyle(
                              fontSize: 12,
                              color: Colors.black54,
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

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
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Text(_status, textAlign: TextAlign.center),
            if (showActions) ...<Widget>[
              const SizedBox(height: 12),
              FilledButton(
                onPressed: () async {
                  if (_cameraPermissionPermanentlyDenied) {
                    await openAppSettings();
                    await _initCamera();
                    return;
                  }
                  await _initCamera();
                },
                child: Text(buttonLabel),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildProgressCard() {
    final ratio = _progress.progressRatio;
    final percent = (ratio * 100).toStringAsFixed(1);
    final completeRatio = _progress.totalSamples <= 0
        ? 0.0
        : (_progress.completeSamples / _progress.totalSamples)
            .clamp(0, 1)
            .toDouble();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFFF1F5F9),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            '进度：${_progress.totalSamples}/${_progress.targetSamples} ($percent%)',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 6),
          LinearProgressIndicator(value: ratio),
          const SizedBox(height: 6),
          Text(
            '完整 33 关键点样本：${_progress.completeSamples}'
            ' (${(completeRatio * 100).toStringAsFixed(1)}%)',
            style: const TextStyle(fontSize: 12, color: Colors.black87),
          ),
          const SizedBox(height: 4),
          Text(
            '动作：${_summaryLine(_progress.byExercise)}',
            style: const TextStyle(fontSize: 12, color: Colors.black54),
          ),
          Text(
            '视角：${_summaryLine(_progress.byView)}',
            style: const TextStyle(fontSize: 12, color: Colors.black54),
          ),
          Text(
            '质量：${_summaryLine(_progress.byQuality)}',
            style: const TextStyle(fontSize: 12, color: Colors.black54),
          ),
          Text(
            '受试者数量：${_progress.bySubject.length}',
            style: const TextStyle(fontSize: 12, color: Colors.black54),
          ),
          const Text(
            '采集模式：单目',
            style: TextStyle(fontSize: 12, color: Colors.black54),
          ),
        ],
      ),
    );
  }

  String _summaryLine(Map<String, int> map) {
    if (map.isEmpty) return '-';
    final entries = map.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return entries
        .take(4)
        .map((e) => '${_localizedTag(e.key)}:${e.value}')
        .join('  ');
  }

  Widget _exerciseDropdown() {
    return DropdownButton<ExerciseType>(
      value: _exerciseType,
      onChanged: (v) {
        if (v != null) setState(() => _exerciseType = v);
      },
      items: ExerciseType.values
          .map(
            (e) => DropdownMenuItem<ExerciseType>(
              value: e,
              child: Text('动作：${e.label}'),
            ),
          )
          .toList(),
    );
  }

  Widget _tagDropdown({
    required String value,
    required String label,
    required List<String> items,
    required ValueChanged<String> onChanged,
  }) {
    return DropdownButton<String>(
      value: value,
      onChanged: (v) {
        if (v != null) onChanged(v);
      },
      items: items
          .map(
            (e) => DropdownMenuItem<String>(
              value: e,
              child: Text('$label: ${_localizedTag(e)}'),
            ),
          )
          .toList(),
    );
  }

  String _localizedTag(String value) {
    switch (value) {
      case 'standard':
        return '标准';
      case 'error':
        return '错误';
      case 'front':
        return '正面';
      case 'side':
        return '侧面';
      case 'oblique':
        return '斜侧';
      case 'single':
        return '单目';
      default:
        return value;
    }
  }

  Future<void> _saveSample() async {
    final pose = _pose;
    if (pose == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('未检测到姿态，无法保存样本')),
      );
      return;
    }
    final leftPoseForSave = _manualPoseSnapshot ?? pose;

    setState(() => _saving = true);
    try {
      final filePath = await _datasetService.appendSample(
        pose: leftPoseForSave,
        exerciseType: _exerciseType,
        qualityTag: _qualityTag,
        viewTag: _viewTag,
        subjectTag: _subjectController.text,
        annotatorId: _annotatorController.text,
        manualChecked: _manualChecked,
        profile: _profile,
        depthMode: PoseDepthMode.monocular,
        primaryCameraName: _primaryCamera?.name,
        correctedLandmarks: _correctedLandmarks,
        requireFullLandmarks: true,
        cameraMode: 'single',
      );
      final progress = await _datasetService.loadProgress();

      if (!mounted) return;
      setState(() {
        _saving = false;
        _lastSavedPath = filePath;
        _progress = progress;
        _correctedLandmarks = null;
        _manualPoseSnapshot = null;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('样本已保存到 JSONL')),
      );
    } on DatasetValidationException catch (e) {
      if (!mounted) return;
      setState(() => _saving = false);
      final missingPreview = e.missingLandmarks.take(6).join(', ');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '保存失败：${e.message}${missingPreview.isEmpty ? '' : ' 缺失：$missingPreview'}',
          ),
        ),
      );
    } catch (_) {
      if (!mounted) return;
      setState(() => _saving = false);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('保存失败：发生未知错误')),
      );
    }
  }

  Future<void> _exportGuideline() async {
    try {
      final filePath = await _datasetService.exportGuideline();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('标注规范已导出：$filePath')),
      );
    } catch (_) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('导出标注规范失败')),
      );
    }
  }

  Widget _buildAdaptiveCameraPreview(CameraController controller) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final previewSize = controller.value.previewSize;
        if (previewSize == null) {
          return const Center(child: CircularProgressIndicator());
        }

        final previewAspectRatio = controller.value.aspectRatio;
        final maxWidth = constraints.maxWidth;
        final maxHeight = constraints.maxHeight;

        var displayWidth = maxWidth;
        var displayHeight = displayWidth / previewAspectRatio;
        if (displayHeight > maxHeight) {
          displayHeight = maxHeight;
          displayWidth = displayHeight * previewAspectRatio;
        }

        final displaySize = Size(displayWidth, displayHeight);
        return Center(
          child: SizedBox(
            width: displayWidth,
            height: displayHeight,
            child: Stack(
              fit: StackFit.expand,
              children: <Widget>[
                CameraPreview(controller),
                if (_pose != null)
                  CustomPaint(
                    painter: PosePainter(
                      pose: _pose!,
                      imageSize: Size(previewSize.height, previewSize.width),
                      canvasSize: displaySize,
                      mirrorX:
                          _cameraLensDirection == CameraLensDirection.front,
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _openLeftCorrection() async {
    final pose = _pose;
    if (pose == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('没有可编辑的关键点')),
      );
      return;
    }

    final result =
        await Navigator.of(context).push<Map<PoseLandmarkType, PoseLandmark>>(
      MaterialPageRoute<Map<PoseLandmarkType, PoseLandmark>>(
        builder: (_) => AnnotationCorrectionScreen(
          title: '关键点修正',
          autoLandmarks: pose.landmarks,
          initialLandmarks: _correctedLandmarks,
        ),
      ),
    );
    if (result == null || !mounted) return;
    setState(() {
      _correctedLandmarks = result;
      _manualPoseSnapshot = pose;
    });
  }
}

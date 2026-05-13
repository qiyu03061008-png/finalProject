import 'package:flutter/material.dart';

import '../models/pose_landmark.dart';

class AnnotationCorrectionScreen extends StatefulWidget {
  const AnnotationCorrectionScreen({
    super.key,
    required this.title,
    required this.autoLandmarks,
    this.initialLandmarks,
  });

  final String title;
  final Map<PoseLandmarkType, PoseLandmark> autoLandmarks;
  final Map<PoseLandmarkType, PoseLandmark>? initialLandmarks;

  @override
  State<AnnotationCorrectionScreen> createState() =>
      _AnnotationCorrectionScreenState();
}

class _AnnotationCorrectionScreenState extends State<AnnotationCorrectionScreen> {
  late Map<PoseLandmarkType, PoseLandmark> _editedLandmarks;
  late PoseLandmarkType _selectedType;

  final _xController = TextEditingController();
  final _yController = TextEditingController();
  final _zController = TextEditingController();
  final _likelihoodController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _editedLandmarks = Map<PoseLandmarkType, PoseLandmark>.from(
      widget.initialLandmarks ?? widget.autoLandmarks,
    );
    _selectedType = _pickInitialType();
    _loadSelectedPoint();
  }

  @override
  void dispose() {
    _xController.dispose();
    _yController.dispose();
    _zController.dispose();
    _likelihoodController.dispose();
    super.dispose();
  }

  PoseLandmarkType _pickInitialType() {
    if (_editedLandmarks.isEmpty) {
      return PoseLandmarkType.nose;
    }
    return _editedLandmarks.keys.first;
  }

  void _loadSelectedPoint() {
    final point = _editedLandmarks[_selectedType];
    if (point == null) {
      _xController.text = '0';
      _yController.text = '0';
      _zController.text = '0';
      _likelihoodController.text = '0';
      return;
    }
    _xController.text = point.x.toStringAsFixed(2);
    _yController.text = point.y.toStringAsFixed(2);
    _zController.text = point.z.toStringAsFixed(2);
    _likelihoodController.text = point.likelihood.toStringAsFixed(3);
  }

  @override
  Widget build(BuildContext context) {
    final editedCount = _editedLandmarks.length;
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: <Widget>[
          TextButton(
            onPressed: _resetToAuto,
            child: const Text('重置'),
          ),
          TextButton(
            onPressed: _applyAndClose,
            child: const Text('完成'),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: <Widget>[
          Text(
            '已编辑关键点：$editedCount/${PoseLandmarkType.values.length}',
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<PoseLandmarkType>(
            value: _selectedType,
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              labelText: '关键点',
            ),
            items: PoseLandmarkType.values
                .map(
                  (type) => DropdownMenuItem<PoseLandmarkType>(
                    value: type,
                    child: Text(type.name),
                  ),
                )
                .toList(),
            onChanged: (type) {
              if (type == null) return;
              setState(() {
                _selectedType = type;
                _loadSelectedPoint();
              });
            },
          ),
          const SizedBox(height: 12),
          _numberField(_xController, 'X 坐标'),
          const SizedBox(height: 10),
          _numberField(_yController, 'Y 坐标'),
          const SizedBox(height: 10),
          _numberField(_zController, 'Z 坐标'),
          const SizedBox(height: 10),
          _numberField(_likelihoodController, '置信度（0~1）'),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: () {
              _applyCurrentPoint();
            },
            icon: const Icon(Icons.check_circle_outline),
            label: const Text('应用当前关键点'),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: _copyFromAutoForCurrent,
            icon: const Icon(Icons.restore),
            label: const Text('恢复此关键点'),
          ),
          const SizedBox(height: 12),
          const Text(
            '提示：置信度通常应在 [0, 1] 范围内。',
            style: TextStyle(fontSize: 12, color: Colors.black54),
          ),
        ],
      ),
    );
  }

  Widget _numberField(TextEditingController controller, String label) {
    return TextField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      decoration: InputDecoration(
        border: const OutlineInputBorder(),
        labelText: label,
      ),
    );
  }

  void _resetToAuto() {
    setState(() {
      _editedLandmarks = Map<PoseLandmarkType, PoseLandmark>.from(
        widget.autoLandmarks,
      );
      _loadSelectedPoint();
    });
  }

  void _copyFromAutoForCurrent() {
    final autoPoint = widget.autoLandmarks[_selectedType];
    if (autoPoint == null) {
      _editedLandmarks.remove(_selectedType);
    } else {
      _editedLandmarks[_selectedType] = autoPoint;
    }
    _loadSelectedPoint();
    setState(() {});
  }

  bool _applyCurrentPoint() {
    final x = double.tryParse(_xController.text);
    final y = double.tryParse(_yController.text);
    final z = double.tryParse(_zController.text);
    final likelihood = double.tryParse(_likelihoodController.text);
    if (x == null || y == null || z == null || likelihood == null) {
      _showError('请输入有效的数字');
      return false;
    }

    if (likelihood < 0 || likelihood > 1) {
      _showError('置信度必须在 0 到 1 之间');
      return false;
    }

    setState(() {
      _editedLandmarks[_selectedType] = PoseLandmark(
        x: x,
        y: y,
        z: z,
        likelihood: likelihood,
      );
    });
    return true;
  }

  void _applyAndClose() {
    if (!_applyCurrentPoint()) {
      return;
    }
    Navigator.of(context).pop<Map<PoseLandmarkType, PoseLandmark>>(
      Map<PoseLandmarkType, PoseLandmark>.from(_editedLandmarks),
    );
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }
}

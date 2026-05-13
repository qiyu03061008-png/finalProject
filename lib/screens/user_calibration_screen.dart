import 'package:flutter/material.dart';

import '../models/user_profile.dart';
import '../services/user_profile_repository.dart';

class UserCalibrationScreen extends StatefulWidget {
  const UserCalibrationScreen({super.key});

  @override
  State<UserCalibrationScreen> createState() => _UserCalibrationScreenState();
}

class _UserCalibrationScreenState extends State<UserCalibrationScreen> {
  final _formKey = GlobalKey<FormState>();
  final _repo = UserProfileRepository();

  final _heightController = TextEditingController();
  final _weightController = TextEditingController();
  final _legRatioController = TextEditingController();
  final _armRatioController = TextEditingController();
  final _shoulderHipController = TextEditingController();
  String? _selectedGender;

  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final profile = await _repo.loadOrDefault();
    _heightController.text = profile.heightCm.toStringAsFixed(1);
    _weightController.text = profile.weightKg.toStringAsFixed(1);
    _legRatioController.text = profile.legLengthRatio.toStringAsFixed(2);
    _armRatioController.text = profile.armLengthRatio.toStringAsFixed(2);
    _shoulderHipController.text = profile.shoulderToHipRatio.toStringAsFixed(2);
    _selectedGender = profile.gender;
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _heightController.dispose();
    _weightController.dispose();
    _legRatioController.dispose();
    _armRatioController.dispose();
    _shoulderHipController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('身体标定')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: <Widget>[
            const Text(
              '首次设置',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
            ),
            const SizedBox(height: 8),
            const Text(
              '这些参数将用于动作阈值自适应和个性化评分。',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 18),
            _numberField(_heightController, '身高（cm）', 100, 230),
            _numberField(_weightController, '体重（kg）', 25, 200),
            _numberField(_legRatioController, '腿长比例（0.45~0.60）', 0.35, 0.7),
            _numberField(_armRatioController, '臂长比例（0.35~0.55）', 0.3, 0.7),
            _numberField(
              _shoulderHipController,
              '肩宽/胯宽比例（0.9~1.5）',
              0.7,
              1.8,
            ),
            const SizedBox(height: 12),
            const Text(
              '性别',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Row(
              children: <Widget>[
                Expanded(
                  child: RadioListTile<String?>(
                    title: const Text('男'),
                    value: 'male',
                    groupValue: _selectedGender,
                    onChanged: (value) {
                      setState(() {
                        _selectedGender = value;
                      });
                    },
                  ),
                ),
                Expanded(
                  child: RadioListTile<String?>(
                    title: const Text('女'),
                    value: 'female',
                    groupValue: _selectedGender,
                    onChanged: (value) {
                      setState(() {
                        _selectedGender = value;
                      });
                    },
                  ),
                ),
                Expanded(
                  child: RadioListTile<String?>(
                    title: const Text('不设置'),
                    value: null,
                    groupValue: _selectedGender,
                    onChanged: (value) {
                      setState(() {
                        _selectedGender = value;
                      });
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 22),
            FilledButton.icon(
              onPressed: _saving ? null : _save,
              icon: _saving
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.save_outlined),
              label: Text(_saving ? '保存中...' : '保存并应用'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _numberField(
    TextEditingController controller,
    String label,
    num min,
    num max,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: controller,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
        ),
        validator: (value) {
          final v = double.tryParse(value ?? '');
          if (v == null) return '请输入数字';
          if (v < min || v > max) return '有效范围：$min ~ $max';
          return null;
        },
      ),
    );
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _saving = true);

    final profile = UserProfile(
      heightCm: double.parse(_heightController.text),
      weightKg: double.parse(_weightController.text),
      legLengthRatio: double.parse(_legRatioController.text),
      armLengthRatio: double.parse(_armRatioController.text),
      shoulderToHipRatio: double.parse(_shoulderHipController.text),
      gender: _selectedGender,
      createdAtIso: DateTime.now().toIso8601String(),
    );
    await _repo.save(profile);

    if (!mounted) return;
    setState(() => _saving = false);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('标定已保存')),
    );
  }
}

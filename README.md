# Fitness Pose App

一个面向毕业设计/课程项目的 Flutter 健身姿态分析应用。当前项目已经不只是“相机识别动作”，而是包含了三条完整链路：

1. 实时训练链路：相机取流 -> 姿态检测 -> 动作分析 -> 实时反馈
2. 个性化链路：用户身体标定 -> 个体化阈值调整 -> 更贴近体型的动作判定
3. 数据闭环链路：样本采集 -> 数据合并 -> 阈值资产生成 -> App 重新加载使用

## 当前真实功能

### 1. 实时训练

- 支持 `深蹲`、`俯卧撑`、`平板支撑`
- 主检测器为 `BlazePose`，可在运行中切换到 `MoveNet`
- 支持单目近似 3D 深度估计
- 当前训练链路采用单目近似 3D 深度估计
- 提供骨架叠加、动作计数、评分、错误提示、性能指标展示

### 2. 个性化标定

- 支持录入身高、体重、腿长比例、臂长比例、肩胯比例、性别
- `PoseAnalyzer` 会结合 `assets/threshold_profile.json` 与用户身体参数动态调整阈值

### 3. 数据集采集

- 支持在 App 内采集姿态样本并保存为 JSONL
- 支持人工修正关键点
- 支持记录动作类型、视角、质量标签、受试者、标注员等信息
- 支持导出标注规范文件

### 4. 离线阈值生成

- 支持将 App 采集数据与 FIT-COACH 数据整理为统一格式
- 支持离线生成新的 `assets/threshold_profile.json`
- 当前训练产物是“阈值配置资产”，不是新的动作分类模型

## 项目结构

```text
lib/
  main.dart                         应用入口
  models/                           数据模型与阈值模型
  screens/
    home_screen.dart                首页入口
    pose_detection_screen.dart      实时训练页
    user_calibration_screen.dart    用户标定页
    dataset_collection_screen.dart  数据采集页
    annotation_correction_screen.dart 关键点人工修正页
  services/
    blazepose_detector.dart         ML Kit BlazePose 检测
    tflite_pose_detector.dart       MoveNet TFLite 检测
    pose_depth_estimator.dart       深度估计
    pose_metric_calculator.dart     姿态指标计算
    pose_analyzer.dart              动作分析与评分
    dataset_collection_service.dart 数据集落盘
    threshold_config_service.dart   阈值资产加载
    user_profile_repository.dart    用户配置存储
    audio_cue_service.dart          语音提示播放
  widgets/
    pose_painter.dart               骨架绘制

scripts/
  sync_collected_dataset.dart       合并采集数据
  generate_thresholds.dart          生成阈值配置
  train_threshold_pipeline.ps1      一键离线流水线

ml/src/
  convert_fitcoach_labels.py
  convert_fitcoach_long_range_feedbacks.py
  extract_fitcoach_landmarks.py
  filter_fitcoach_to_available_videos.py
```

## 运行环境

### Flutter 侧

- Flutter 3.x
- Dart 3.x
- Android 真机优先

### Python 侧

- 推荐 `Python 3.11`
- `ml/requirements_extract.txt` 中的依赖用于离线数据处理

## 快速开始

### 1. 安装依赖

```bash
flutter pub get
```

### 2. 运行 App

```bash
flutter run
```

### 3. 可选：准备 MoveNet 模型

仓库当前已经包含：

- `assets/models/movenet_lightning.tflite`

如果后续替换模型，可参考：

- `assets/models/README.md`
- `download_model.bat`

## 真实运行链路

### 实时训练链路

1. `PoseDetectionScreen` 打开相机并开始图像流
2. `BlazePoseDetector` 或 `TFLitePoseDetector` 输出关键点
3. `PoseDepthEstimator` 生成单目近似 3D 深度结果
4. `PoseMetricCalculator` 计算角度、偏移、身体轴线等指标
5. `PoseAnalyzer` 结合用户画像和阈值配置输出评分、计数和错误提示
6. `PosePainter` 绘制骨架
7. `AudioCueService` 按分析结果播放语音提示

### 离线阈值链路

1. App 采集数据写入本地 JSONL
2. `scripts/sync_collected_dataset.dart` 合并数据到 `data/dataset/fitness_dataset.jsonl`
3. `scripts/generate_thresholds.dart` 统计动作指标并生成阈值
4. 输出覆盖 `assets/threshold_profile.json`
5. App 启动时由 `ThresholdConfigService` 加载该资产

## 离线流水线命令

### 仅合并采集数据

```bash
dart run scripts/sync_collected_dataset.dart
```

### 仅生成阈值资产

```bash
dart run scripts/generate_thresholds.dart
```

### 一键跑完整流水线

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_threshold_pipeline.ps1 -PythonExe .\.venv_ml311\Scripts\python.exe
```

更多说明见 [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)。

## 资源文件

- `assets/threshold_profile.json`
  - App 启动时加载的核心阈值资产
- `assets/models/movenet_lightning.tflite`
  - MoveNet 检测模型
- `assets/audio/`
  - 语音提示资源目录

## 当前已知限制

### 1. 数据采集页当前采用单目采集

当前 `DatasetCollectionScreen` 已收敛为单目采集流程，数据采集链路为“单目采集 + 手工修正 + JSONL 导出”。



### 3. 当前项目的“训练”是阈值训练，不是端到端模型训练

离线流程输出的是阈值配置，而不是新的 `.tflite` 动作质量模型。

## 更适合写进答辩/论文的点

- 基于移动端的实时姿态检测与动作纠错
- 规则分析与个体化阈值的结合
- 数据采集、标注修正、阈值回流形成闭环
- 双检测器切换：`BlazePose` 与 `MoveNet`
- 单目近似 3D 深度估计的实现尝试

## 建议的下一步

- 继续完善数据采集体验，例如样本筛选、批量管理与导出流程
- 补齐 `assets/audio/` 里的真实语音资源
- 记录训练历史与会话报表，形成可视化结果页
- 增加更多自动化测试，尤其是 `PoseAnalyzer` 和数据脚本

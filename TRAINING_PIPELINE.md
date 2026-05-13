# Training Pipeline

这个项目当前的“训练”不是训练一个新的动作识别模型，而是把你的标注数据集转换成统一 JSONL，再生成 App 运行时要使用的阈值资产。

## 真实运行链路

1. `assets/models/movenet_lightning.tflite` 负责人体关键点检测。
2. `lib/services/tflite_pose_detector.dart` 调用这个 TFLite 模型做推理。
3. `lib/services/pose_analyzer.dart` 根据关键点计算角度、躯干倾斜、膝盖内扣等指标。
4. `assets/threshold_profile.json` 决定这些指标的判定阈值。
5. 你的离线训练流程，本质上是在更新 `assets/threshold_profile.json`。

所以：

- 你当前项目的主要训练产物是 `assets/threshold_profile.json`
- 不是导出一个新的动作质量分类模型
- 后续功能当前不是依赖“你自己训练出的新模型出口”
- 而是依赖 “MoveNet + 阈值配置 + PoseAnalyzer”

## 什么时候才需要模型导出

只有当你想把当前规则分析器换成“端到端模型判断动作质量”时，才需要训练并导出新的 `.tflite` / `.onnx` 模型，并修改 Flutter 端推理逻辑。

按你现在这套代码，后续功能不应该主要通过“新模型出口”完成，而应该通过更新阈值资产完成。

## 你现在应该跑的命令

### 1. 安装离线提取依赖

如果你当前 `python` 是 3.13，并且曾经报过：

```text
AttributeError: module 'mediapipe' has no attribute 'solutions'
```

说明你当前的 MediaPipe 构建不兼容现有提取脚本。推荐直接切到 Python 3.11。

先安装 Python 3.11，然后执行：

```powershell
py -3.11 -m venv .venv_ml311
.\.venv_ml311\Scripts\Activate.ps1
python -m pip install -r ml/requirements_extract.txt
```

如果你暂时不想激活环境，也可以后面直接显式指定解释器路径。

### 2. 一键跑完整训练管线

推荐这样跑：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_threshold_pipeline.ps1 -PythonExe .\.venv_ml311\Scripts\python.exe
```

这条命令会按顺序尝试：

1. 生成 FIT-COACH 映射标签
2. 提取 FIT-COACH landmarks
3. 合并所有训练集 JSONL
4. 生成 `assets/threshold_profile.json`

### 3. 手动逐步跑

#### FIT-COACH 标签映射

```powershell
.\.venv_ml311\Scripts\python.exe ml/src/convert_fitcoach_labels.py `
  --input data/qevd_fitcoach/extracted/QEVD-FIT-COACH/fine_grained_labels.json `
  --feedback-short data/qevd_fitcoach/extracted/QEVD-FIT-COACH/feedbacks_short_clips.json `
  --feedback-long data/qevd_fitcoach/extracted/QEVD-FIT-COACH/feedbacks_long_range.json `
  --output data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl `
  --report data/qevd_fitcoach/processed/fitcoach_mapped_app.report.json `
  --app-only
```

#### FIT-COACH landmarks 提取

```powershell
.\.venv_ml311\Scripts\python.exe ml/src/extract_fitcoach_landmarks.py `
  --input-jsonl data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl `
  --video-root data/qevd_fitcoach/extracted/QEVD-FIT-COACH `
  --output data/qevd_fitcoach/processed/fitcoach_landmarks_app.jsonl `
  --report data/qevd_fitcoach/processed/fitcoach_landmarks_app.report.json `
  --resume
```

#### 合并训练集

```powershell
dart run scripts/sync_collected_dataset.dart
```

#### 生成最终阈值资产

```powershell
dart run scripts/generate_thresholds.dart
```

## 最终产物

离线训练阶段最关键的输出有：

- `data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl`
- `data/qevd_fitcoach/processed/fitcoach_landmarks_app.jsonl`
- `data/dataset/fitness_dataset.jsonl`
- `data/dataset/fitness_dataset_sync.report.json`
- `assets/threshold_profile.json`

其中真正接入 Flutter App 的关键产物是：

- `assets/threshold_profile.json`

## 这些产物如何接入代码

代码里已经直接读取：

- `lib/services/threshold_config_service.dart`

运行时分析走：

- `lib/services/pose_analyzer.dart`

人体关键点检测走：

- `lib/services/tflite_pose_detector.dart`

所以你训练完成后，只要重新运行 App，新阈值就会进入整条分析链路。

## 当前系统的边界

你现在训练出来的是“阈值资产”，不是“新模型”。

如果你后面想做真正的模型训练与导出，通常要新增：

1. 样本切片与标签标准化
2. 时序模型训练代码
3. 模型导出为 `.tflite`
4. Flutter 端新的推理服务
5. 替换或并行保留 `PoseAnalyzer`

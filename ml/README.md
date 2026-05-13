# ML/Data Tools

当前 `ml/` 目录只保留和“数据整理 / 阈值统计输入准备”直接相关的工具。

## 还在使用的脚本

- `ml/src/convert_fitcoach_labels.py`
  - 把 FIT-COACH 原始标签映射成 App 当前使用的动作/错误类型 schema
- `ml/src/extract_fitcoach_landmarks.py`
  - 批量从 FIT-COACH 视频中抽取 BlazePose 33 关键点
  - 输出工程可直接消费的 JSONL，供阈值统计使用

## 依赖安装

只需要安装抽关键点所需依赖：

```bash
pip install -r ml/requirements_extract.txt
```

## 推荐流程

1. 先生成 FIT-COACH 标签映射：

```bash
python ml/src/convert_fitcoach_labels.py \
  --input data/qevd_fitcoach/extracted/QEVD-FIT-COACH/fine_grained_labels.json \
  --feedback-short data/qevd_fitcoach/extracted/QEVD-FIT-COACH/feedbacks_short_clips.json \
  --feedback-long data/qevd_fitcoach/extracted/QEVD-FIT-COACH/feedbacks_long_range.json \
  --output data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl \
  --report data/qevd_fitcoach/processed/fitcoach_mapped_app.report.json \
  --app-only
```

2. 再批量抽取 landmarks：

```bash
python ml/src/extract_fitcoach_landmarks.py --resume
```

3. 最后回到主链路生成阈值：

```bash
dart run scripts/sync_collected_dataset.dart
dart run scripts/generate_thresholds.dart
```


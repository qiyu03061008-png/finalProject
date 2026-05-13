# MoveNet模型文件

请将下载的 `movenet_lightning.tflite` 文件放置在此目录。

## 下载方法

### 方法1: 使用下载脚本（推荐）
在项目根目录运行：
```bash
download_model.bat
```

### 方法2: 手动下载
1. 访问: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4
2. 下载模型文件
3. 重命名为: movenet_lightning.tflite
4. 放置到此目录

### 方法3: 使用PowerShell命令
```powershell
Invoke-WebRequest -Uri 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite' -OutFile 'movenet_lightning.tflite'
```

## 文件信息
- 文件名: movenet_lightning.tflite
- 大小: 约3MB
- 格式: TensorFlow Lite (int8量化)
- 输入: 192x192 RGB图像
- 输出: 17个关键点

## 验证
确保文件路径为：
```
fitness_pose_app/assets/models/movenet_lightning.tflite
```

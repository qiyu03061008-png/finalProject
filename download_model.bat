@echo off
echo ========================================
echo MoveNet模型下载脚本
echo ========================================
echo.

REM 创建assets目录
if not exist "assets\models" (
    echo 创建assets\models目录...
    mkdir assets\models
)

echo.
echo 正在下载MoveNet Lightning模型...
echo.

REM 使用PowerShell下载
powershell -Command "Invoke-WebRequest -Uri 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite' -OutFile 'assets\models\movenet_lightning.tflite'"

if exist "assets\models\movenet_lightning.tflite" (
    echo.
    echo ========================================
    echo 模型下载成功！
    echo 文件位置: assets\models\movenet_lightning.tflite
    for %%A in ("assets\models\movenet_lightning.tflite") do echo 文件大小: %%~zA 字节
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 下载失败！
    echo.
    echo 请手动下载：
    echo 1. 访问: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4
    echo 2. 点击下载按钮
    echo 3. 将文件重命名为: movenet_lightning.tflite
    echo 4. 放置到: assets\models\ 目录
    echo ========================================
)

echo.
pause

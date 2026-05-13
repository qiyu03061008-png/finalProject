@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "DEEP_CLEAN=0"
if /I "%~1"=="--deep-clean" set "DEEP_CLEAN=1"
if /I "%~1"=="--clean" set "DEEP_CLEAN=1"

echo ========================================
echo Fitness Pose App - prepare and run
echo ========================================
echo.

if "%DEEP_CLEAN%"=="1" (
    echo [1/3] Running deep clean...
    call flutter clean
    echo.
) else (
    echo [1/3] Keeping Gradle and Flutter caches for faster incremental builds...
    echo       Use clean_and_run.bat --deep-clean only when the build is actually broken.
    echo.
)

echo [2/3] Fetching dependencies...
call flutter pub get
echo.

echo [3/3] Listing devices...
call flutter devices
echo.

echo ========================================
echo Ready
echo ========================================
echo Example:
echo flutter run -d emulator-5554
echo.

pause

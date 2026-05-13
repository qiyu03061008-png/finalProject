param(
  [string]$PythonExe = 'python',
  [switch]$SkipFitCoachMapping,
  [switch]$SkipFitCoachLandmarks,
  [switch]$SkipSync,
  [switch]$SkipThresholds,
  [switch]$ResumeFitCoach = $true,
  [switch]$AllowNumericFallback = $true,
  [int]$FitCoachMaxVideos = 0
)

$ErrorActionPreference = 'Stop'

function Step([string]$message) {
  Write-Host ""
  Write-Host "==> $message" -ForegroundColor Cyan
}

function Assert-CommandExists([string]$name) {
  if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $name"
  }
}

function Assert-PathExists([string]$path, [string]$message) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw $message
  }
}

function Run-Checked([string[]]$commandParts) {
  $commandText = ($commandParts | ForEach-Object {
      if ($_ -match '\s') { '"{0}"' -f $_ } else { $_ }
    }) -join ' '
  Write-Host $commandText -ForegroundColor DarkGray
  & $commandParts[0] $commandParts[1..($commandParts.Length - 1)]
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed with exit code ${LASTEXITCODE}: $commandText"
  }
}

function File-Exists([string]$path) {
  return Test-Path -LiteralPath $path
}

function Dir-HasMp4([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) {
    return $false
  }
  return @(Get-ChildItem -LiteralPath $path -Recurse -File -Filter *.mp4 -ErrorAction SilentlyContinue).Count -gt 0
}

Assert-CommandExists 'python'
Assert-CommandExists 'dart'
if ($PythonExe -ne 'python') {
  Assert-PathExists $PythonExe "Configured Python executable was not found: $PythonExe"
}

$fitCoachRoot = 'data/qevd_fitcoach/extracted/QEVD-FIT-COACH'
$fitCoachLongRangeRoot = 'data/qevd_fitcoach/extracted/QEVD-FIT-COACH/long_range_videos'
$fitCoachMapped = 'data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl'
$fitCoachLongRangeJsonl = 'data/qevd_fitcoach/processed/fitcoach_long_range_app.jsonl'
$fitCoachLandmarks = 'data/qevd_fitcoach/processed/fitcoach_landmarks_app.jsonl'

Step 'Checking extraction dependencies'
if (-not (File-Exists 'ml/requirements_extract.txt')) {
  throw 'Missing ml/requirements_extract.txt'
}
Write-Host "Training Python: $PythonExe"
Write-Host "Install once if needed: $PythonExe -m pip install -r ml/requirements_extract.txt"

Step 'Checking MediaPipe compatibility'
$mpCheck = @'
import sys
try:
    import mediapipe as mp
except Exception as exc:
    print(f"MEDIAPIPE_IMPORT_ERROR::{type(exc).__name__}::{exc}")
    raise SystemExit(2)

has_solutions = hasattr(mp, "solutions")
print(f"MEDIAPIPE_OK::{sys.version.split()[0]}::{getattr(mp, '__file__', 'unknown')}::{has_solutions}")
if not has_solutions:
    raise SystemExit(3)
'@
$mpTemp = Join-Path $PWD '.tmp_mediapipe_check.py'
Set-Content -LiteralPath $mpTemp -Value $mpCheck -Encoding ASCII
try {
  $mpOutput = & $PythonExe $mpTemp 2>&1
  $mpExit = $LASTEXITCODE
} finally {
  Remove-Item -LiteralPath $mpTemp -Force -ErrorAction SilentlyContinue
}
if ($mpOutput) {
  $mpOutput | ForEach-Object { Write-Host $_ }
}
if ($mpExit -eq 2) {
  throw "MediaPipe could not be imported by $PythonExe. Run: $PythonExe -m pip install -r ml/requirements_extract.txt"
}
if ($mpExit -eq 3) {
  throw @"
The current Python interpreter is using a MediaPipe build without mediapipe.solutions.
Your current environment is not compatible with ml/src/extract_fitcoach_landmarks.py.

Recommended fix:
1. Install Python 3.11
2. Create a new venv:
   py -3.11 -m venv .venv_ml311
3. Activate it:
   .\.venv_ml311\Scripts\Activate.ps1
4. Install extraction deps:
   python -m pip install -r ml/requirements_extract.txt
5. Re-run with:
   powershell -ExecutionPolicy Bypass -File .\scripts\train_threshold_pipeline.ps1 -PythonExe .\.venv_ml311\Scripts\python.exe
"@
}
if ($mpExit -ne 0) {
  throw "Unexpected MediaPipe compatibility check failure with exit code $mpExit"
}

if (-not $SkipFitCoachMapping) {
  Step 'Building FIT-COACH mapped labels'
  if (
    (File-Exists "$fitCoachRoot/fine_grained_labels.json") -and
    (File-Exists "$fitCoachRoot/feedbacks_short_clips.json") -and
    (File-Exists "$fitCoachRoot/feedbacks_long_range.json")
  ) {
    Run-Checked @(
      $PythonExe,
      'ml/src/convert_fitcoach_labels.py',
      '--input', "$fitCoachRoot/fine_grained_labels.json",
      '--feedback-short', "$fitCoachRoot/feedbacks_short_clips.json",
      '--feedback-long', "$fitCoachRoot/feedbacks_long_range.json",
      '--output', $fitCoachMapped,
      '--report', 'data/qevd_fitcoach/processed/fitcoach_mapped_app.report.json',
      '--app-only'
    )
  } else {
    Write-Warning 'Skipped FIT-COACH mapping because required extracted label files are missing.'
  }
}

if (-not $SkipFitCoachLandmarks) {
  Step 'Extracting FIT-COACH landmarks'
  if (Dir-HasMp4 $fitCoachLongRangeRoot) {
    Run-Checked @(
      $PythonExe,
      'ml/src/convert_fitcoach_long_range_feedbacks.py',
      '--input', "$fitCoachRoot/feedbacks_long_range.json",
      '--output', $fitCoachLongRangeJsonl,
      '--report', 'data/qevd_fitcoach/processed/fitcoach_long_range_app.report.json'
    )
    $fitCoachArgs = @(
      $PythonExe,
      'ml/src/extract_fitcoach_landmarks.py',
      '--input-jsonl', $fitCoachLongRangeJsonl,
      '--video-root', $fitCoachLongRangeRoot,
      '--output', $fitCoachLandmarks,
      '--report', 'data/qevd_fitcoach/processed/fitcoach_landmarks_app.report.json'
    )
    if ($ResumeFitCoach) {
      $fitCoachArgs += '--resume'
    }
    if ($AllowNumericFallback) {
      $fitCoachArgs += '--allow-numeric-fallback'
    }
    if ($FitCoachMaxVideos -gt 0) {
      $fitCoachArgs += @('--max-videos', "$FitCoachMaxVideos")
    }
    Run-Checked $fitCoachArgs
  } else {
    Write-Warning 'Skipped FIT-COACH landmark extraction because long_range_videos mp4 files are missing.'
  }
}

if (-not $SkipSync) {
  Step 'Merging dataset JSONL files'
  Run-Checked @(
    'dart',
    'run',
    'scripts/sync_collected_dataset.dart'
  )
}

if (-not $SkipThresholds) {
  Step 'Generating threshold profile asset'
  Run-Checked @(
    'dart',
    'run',
    'scripts/generate_thresholds.dart'
  )
}

Step 'Training pipeline finished'
Write-Host 'Main output asset: assets/threshold_profile.json' -ForegroundColor Green
Write-Host 'Merged dataset: data/dataset/fitness_dataset.jsonl' -ForegroundColor Green

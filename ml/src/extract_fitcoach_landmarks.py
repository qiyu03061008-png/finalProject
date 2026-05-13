from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime with a friendly error.
    np = None


LANDMARK_NAMES = [
    "nose",
    "leftEyeInner",
    "leftEye",
    "leftEyeOuter",
    "rightEyeInner",
    "rightEye",
    "rightEyeOuter",
    "leftEar",
    "rightEar",
    "leftMouth",
    "rightMouth",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftPinky",
    "rightPinky",
    "leftIndex",
    "rightIndex",
    "leftThumb",
    "rightThumb",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
    "leftHeel",
    "rightHeel",
    "leftFootIndex",
    "rightFootIndex",
]

CORE_LANDMARK_NAMES = [
    "leftShoulder",
    "rightShoulder",
    "leftHip",
    "rightHip",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

APP_EXERCISES = {"squat", "pushup", "plank"}
EXERCISE_TO_PHASE_METRIC = {
    "squat": "kneeAngleDeg",
    "pushup": "elbowAngleDeg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch extract BlazePose-style landmarks from FIT-COACH videos and "
            "write records that match the app JSONL schema."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl",
        help="Input mapped FIT-COACH JSONL path.",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="data/qevd_fitcoach/extracted/QEVD-FIT-COACH",
        help="Root directory that contains FIT-COACH videos.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_landmarks_app.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_landmarks_app.report.json",
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=3,
        help="How many representative frames to keep per video.",
    )
    parser.add_argument(
        "--probe-frames",
        type=int,
        default=18,
        help="How many frames to sample from each video before selecting representatives.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.45,
        help="Minimum landmark visibility/presence used for quality scoring.",
    )
    parser.add_argument(
        "--min-core-score",
        type=float,
        default=0.55,
        help="Minimum average score across core joints.",
    )
    parser.add_argument(
        "--min-detected-points",
        type=int,
        default=24,
        help="Minimum number of reliable landmarks required for a usable frame.",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="MediaPipe Pose model complexity. 0 is fastest; 2 is slowest.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe minimum tracking confidence.",
    )
    parser.add_argument(
        "--exercise",
        type=str,
        default="",
        help="Optional comma-separated filter, e.g. squat,pushup",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Optional cap for smoke tests. 0 means no cap.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip source records already present in the output JSONL.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output JSONL before writing.",
    )
    parser.add_argument(
        "--allow-numeric-fallback",
        action="store_true",
        help=(
            "Allow matching 00000042.mp4 to numeric variants like 0042.mp4. "
            "Keep this off unless you are sure the filenames refer to the same clip."
        ),
    )
    parser.add_argument(
        "--rotate",
        type=str,
        choices=["none", "cw", "ccw", "180"],
        default="none",
        help=(
            "Rotate every decoded video frame before pose extraction. "
            "Use this when OpenCV reads portrait videos sideways."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class FrameObservation:
    frame_index: int
    time_sec: float
    frame_width: int
    frame_height: int
    view_tag: str
    quality_score: float
    core_score: float
    reliable_landmark_count: int
    landmarks: dict[str, dict[str, float]]
    metrics: dict[str, float]


class VideoIndex:
    def __init__(self, video_root: Path) -> None:
        self.video_root = video_root
        self._exact: dict[str, list[Path]] = defaultdict(list)
        self._numeric: dict[str, list[Path]] = defaultdict(list)
        self.video_count = 0

        for path in video_root.rglob("*.mp4"):
            self.video_count += 1
            self._exact[path.name.lower()].append(path)
            numeric = normalize_numeric_stem(path.stem)
            if numeric is not None:
                self._numeric[numeric].append(path)

    def resolve(
        self,
        raw_video_path: str,
        *,
        allow_numeric_fallback: bool = False,
    ) -> tuple[Path | None, str]:
        raw_path = raw_video_path.strip()
        if not raw_path:
            return None, "missing"

        direct_path = (self.video_root / raw_path).resolve()
        if direct_path.exists():
            return direct_path, "relative"

        basename = Path(raw_path).name.lower()
        exact_matches = self._exact.get(basename, [])
        if len(exact_matches) == 1:
            return exact_matches[0], "exact"
        if len(exact_matches) > 1:
            return None, "ambiguous"

        numeric = normalize_numeric_stem(Path(raw_path).stem)
        if allow_numeric_fallback and numeric is not None:
            numeric_matches = self._numeric.get(numeric, [])
            if len(numeric_matches) == 1:
                return numeric_matches[0], "numeric"
            if len(numeric_matches) > 1:
                return None, "ambiguous"

        return None, "missing"


class FitCoachLandmarkExtractor:
    def __init__(
        self,
        *,
        model_complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        min_visibility: float,
        min_core_score: float,
        min_detected_points: int,
        rotate_mode: str,
    ) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is not installed. Please run: pip install -r ml/requirements_extract.txt"
            ) from exc

        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Please run: pip install -r ml/requirements_extract.txt"
            ) from exc

        if not hasattr(mp, "solutions"):
            module_file = getattr(mp, "__file__", "unknown")
            raise RuntimeError(
                "The installed mediapipe package is incompatible with this extractor. "
                f"Expected mediapipe.solutions.pose, but it is missing in: {module_file}\n"
                "This project's landmark extraction scripts currently require the legacy "
                "`mediapipe.solutions` API, which is typically available in a Python 3.10/3.11 "
                "environment with a compatible mediapipe wheel.\n"
                "Recommended fix on Windows:\n"
                "  1. Install Python 3.11\n"
                "  2. Create a clean venv: py -3.11 -m venv .venv_ml311\n"
                "  3. Activate it: .\\.venv_ml311\\Scripts\\Activate.ps1\n"
                "  4. Install deps: python -m pip install -r ml/requirements_extract.txt\n"
                "  5. Re-run the training pipeline with that interpreter."
            )

        self.cv2 = cv2
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.min_visibility = float(min_visibility)
        self.min_core_score = float(min_core_score)
        self.min_detected_points = int(min_detected_points)
        self.rotate_mode = rotate_mode

    def close(self) -> None:
        self._pose.close()

    def extract_video(
        self,
        video_path: Path,
        exercise: str,
        probe_frames: int,
        frames_per_video: int,
        target_time_sec: float | None = None,
    ) -> list[FrameObservation]:
        cap = self.cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        try:
            frame_count = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(self.cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 1e-5:
                fps = 30.0

            candidate_indices = choose_probe_indices(
                frame_count=frame_count,
                probe_frames=probe_frames,
                fps=fps,
                target_time_sec=target_time_sec,
            )
            observations: list[FrameObservation] = []
            for frame_index in candidate_indices:
                cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue

                observation = self._extract_frame(
                    frame_bgr=frame_bgr,
                    frame_index=frame_index,
                    time_sec=frame_index / fps,
                )
                if observation is not None:
                    observations.append(observation)

            return select_representative_frames(
                observations=observations,
                exercise=exercise,
                frames_per_video=frames_per_video,
            )
        finally:
            cap.release()

    def _extract_frame(
        self,
        *,
        frame_bgr: np.ndarray,
        frame_index: int,
        time_sec: float,
    ) -> FrameObservation | None:
        frame_bgr = self._rotate_frame(frame_bgr)
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if result.pose_landmarks is None:
            return None

        image_landmarks = result.pose_landmarks.landmark
        world_landmarks = (
            result.pose_world_landmarks.landmark
            if result.pose_world_landmarks is not None
            else None
        )
        if len(image_landmarks) < len(LANDMARK_NAMES):
            return None

        z_scale = estimate_world_z_scale(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        landmarks = build_landmark_payload(
            image_landmarks=image_landmarks,
            world_landmarks=world_landmarks,
            frame_width=frame_width,
            frame_height=frame_height,
            z_scale=z_scale,
        )

        reliable_scores = [
            landmark["likelihood"]
            for landmark in landmarks.values()
            if landmark["likelihood"] >= self.min_visibility
        ]
        reliable_landmark_count = len(reliable_scores)
        if reliable_landmark_count < self.min_detected_points:
            return None

        core_scores = [
            landmarks[name]["likelihood"]
            for name in CORE_LANDMARK_NAMES
            if name in landmarks
        ]
        if not core_scores:
            return None
        core_score = float(sum(core_scores) / len(core_scores))
        if core_score < self.min_core_score:
            return None

        quality_score = 0.7 * core_score + 0.3 * (
            reliable_landmark_count / float(len(LANDMARK_NAMES))
        )
        metrics = build_frame_metrics(landmarks)

        return FrameObservation(
            frame_index=frame_index,
            time_sec=time_sec,
            frame_width=frame_width,
            frame_height=frame_height,
            view_tag=infer_view_tag(landmarks),
            quality_score=float(quality_score),
            core_score=core_score,
            reliable_landmark_count=reliable_landmark_count,
            landmarks=landmarks,
            metrics=metrics,
        )

    def _rotate_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.rotate_mode == "cw":
            return self.cv2.rotate(frame_bgr, self.cv2.ROTATE_90_CLOCKWISE)
        if self.rotate_mode == "ccw":
            return self.cv2.rotate(frame_bgr, self.cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.rotate_mode == "180":
            return self.cv2.rotate(frame_bgr, self.cv2.ROTATE_180)
        return frame_bgr


def build_landmark_payload(
    *,
    image_landmarks,
    world_landmarks,
    frame_width: int,
    frame_height: int,
    z_scale: float,
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(LANDMARK_NAMES):
        image_lm = image_landmarks[idx]
        score = landmark_score(image_lm)

        z_value = float(getattr(image_lm, "z", 0.0)) * float(frame_width)
        if world_landmarks is not None and idx < len(world_landmarks):
            world_z = float(getattr(world_landmarks[idx], "z", 0.0))
            z_value = world_z * z_scale

        payload[name] = {
            "x": float(image_lm.x) * float(frame_width),
            "y": float(image_lm.y) * float(frame_height),
            "z": float(z_value),
            "likelihood": float(max(0.0, min(1.0, score))),
        }
    return payload


def estimate_world_z_scale(
    *,
    image_landmarks,
    world_landmarks,
    frame_width: int,
    frame_height: int,
) -> float:
    if world_landmarks is None or len(world_landmarks) <= 12:
        return float(frame_width)

    image_left = image_landmarks[11]
    image_right = image_landmarks[12]
    world_left = world_landmarks[11]
    world_right = world_landmarks[12]

    shoulder_px = math.hypot(
        float(image_left.x - image_right.x) * float(frame_width),
        float(image_left.y - image_right.y) * float(frame_height),
    )
    shoulder_world = math.sqrt(
        float(world_left.x - world_right.x) ** 2
        + float(world_left.y - world_right.y) ** 2
        + float(world_left.z - world_right.z) ** 2
    )
    if shoulder_px < 1e-5 or shoulder_world < 1e-5:
        return float(frame_width)
    return shoulder_px / shoulder_world


def landmark_score(landmark) -> float:
    scores: list[float] = []
    visibility = getattr(landmark, "visibility", None)
    presence = getattr(landmark, "presence", None)
    if visibility is not None:
        scores.append(float(visibility))
    # Some MediaPipe builds report presence as 0.0 for every pose landmark even
    # when visibility is high, which would incorrectly reject valid detections.
    if presence is not None and float(presence) > 0.0:
        scores.append(float(presence))
    if not scores:
        return 1.0
    return min(scores)


def choose_probe_indices(
    frame_count: int,
    probe_frames: int,
    fps: float | None = None,
    target_time_sec: float | None = None,
) -> list[int]:
    if probe_frames <= 0:
        return [0]
    if frame_count <= 0:
        return list(range(probe_frames))
    if frame_count == 1:
        return [0]
    if target_time_sec is not None and fps is not None and fps > 1e-5:
        center_index = int(round(target_time_sec * fps))
        center_index = max(0, min(frame_count - 1, center_index))
        half_window = max(2, int(round(fps * 1.5)))
        start = max(0, center_index - half_window)
        end = min(frame_count - 1, center_index + half_window)
        if end <= start:
            return [center_index]
        raw = np.linspace(start, end, num=min(end - start + 1, probe_frames))
        return sorted({int(round(v)) for v in raw.tolist()})
    raw = np.linspace(0, frame_count - 1, num=min(frame_count, probe_frames))
    return sorted({int(round(v)) for v in raw.tolist()})


def select_representative_frames(
    *,
    observations: list[FrameObservation],
    exercise: str,
    frames_per_video: int,
) -> list[FrameObservation]:
    if not observations:
        return []

    frames_per_video = max(1, frames_per_video)
    selected: list[FrameObservation] = []

    metric_name = EXERCISE_TO_PHASE_METRIC.get(exercise)
    if metric_name is not None:
        metric_ready = [o for o in observations if metric_name in o.metrics]
        if metric_ready:
            metric_ready.sort(key=lambda item: item.metrics[metric_name])
            quantile_positions = np.linspace(
                0,
                len(metric_ready) - 1,
                num=min(frames_per_video, len(metric_ready)),
            )
            seen_frames: set[int] = set()
            for position in quantile_positions.tolist():
                candidate = metric_ready[int(round(position))]
                if candidate.frame_index in seen_frames:
                    continue
                selected.append(candidate)
                seen_frames.add(candidate.frame_index)

    if not selected:
        time_sorted = sorted(
            observations,
            key=lambda item: (item.frame_index, -item.quality_score),
        )
        quantile_positions = np.linspace(
            0,
            len(time_sorted) - 1,
            num=min(frames_per_video, len(time_sorted)),
        )
        seen_frames = set()
        for position in quantile_positions.tolist():
            candidate = time_sorted[int(round(position))]
            if candidate.frame_index in seen_frames:
                continue
            selected.append(candidate)
            seen_frames.add(candidate.frame_index)

    if len(selected) < min(frames_per_video, len(observations)):
        already = {item.frame_index for item in selected}
        for candidate in sorted(
            observations,
            key=lambda item: (-item.quality_score, item.frame_index),
        ):
            if candidate.frame_index in already:
                continue
            selected.append(candidate)
            already.add(candidate.frame_index)
            if len(selected) >= min(frames_per_video, len(observations)):
                break

    selected.sort(key=lambda item: item.frame_index)
    return selected[:frames_per_video]


def infer_view_tag(landmarks: dict[str, dict[str, float]]) -> str:
    left_shoulder = landmarks.get("leftShoulder")
    right_shoulder = landmarks.get("rightShoulder")
    left_hip = landmarks.get("leftHip")
    right_hip = landmarks.get("rightHip")

    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return "front"

    if min(
        left_shoulder["likelihood"],
        right_shoulder["likelihood"],
        left_hip["likelihood"],
        right_hip["likelihood"],
    ) < 0.45:
        return "front"

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    torso_length = distance2(shoulder_center, hip_center)
    if torso_length < 1e-5:
        return "front"

    shoulder_width = distance2(left_shoulder, right_shoulder)
    hip_width = distance2(left_hip, right_hip)
    width_ratio = ((shoulder_width + hip_width) / 2.0) / torso_length
    if width_ratio <= 0.32:
        return "side"
    if width_ratio <= 0.60:
        return "oblique"
    return "front"


def build_frame_metrics(landmarks: dict[str, dict[str, float]]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    knee_angles = [
        joint_angle(
            landmarks.get("leftHip"),
            landmarks.get("leftKnee"),
            landmarks.get("leftAnkle"),
        ),
        joint_angle(
            landmarks.get("rightHip"),
            landmarks.get("rightKnee"),
            landmarks.get("rightAnkle"),
        ),
    ]
    knee_angles = [value for value in knee_angles if value is not None]
    if knee_angles:
        metrics["kneeAngleDeg"] = float(sum(knee_angles) / len(knee_angles))

    elbow_angles = [
        joint_angle(
            landmarks.get("leftShoulder"),
            landmarks.get("leftElbow"),
            landmarks.get("leftWrist"),
        ),
        joint_angle(
            landmarks.get("rightShoulder"),
            landmarks.get("rightElbow"),
            landmarks.get("rightWrist"),
        ),
    ]
    elbow_angles = [value for value in elbow_angles if value is not None]
    if elbow_angles:
        metrics["elbowAngleDeg"] = float(sum(elbow_angles) / len(elbow_angles))

    left_body_line = joint_angle(
        landmarks.get("leftShoulder"),
        landmarks.get("leftHip"),
        landmarks.get("leftAnkle"),
    )
    right_body_line = joint_angle(
        landmarks.get("rightShoulder"),
        landmarks.get("rightHip"),
        landmarks.get("rightAnkle"),
    )
    body_line = average_optional(left_body_line, right_body_line)
    if body_line is not None:
        metrics["bodyLineAngleDeg"] = body_line
        metrics["bodyLineDeviationDeg"] = abs(180.0 - body_line)

    return metrics


def joint_angle(
    point_a: dict[str, float] | None,
    point_b: dict[str, float] | None,
    point_c: dict[str, float] | None,
) -> float | None:
    if point_a is None or point_b is None or point_c is None:
        return None
    if min(point_a["likelihood"], point_b["likelihood"], point_c["likelihood"]) < 0.30:
        return None

    vec_ba = np.array(
        [
            point_a["x"] - point_b["x"],
            point_a["y"] - point_b["y"],
            point_a["z"] - point_b["z"],
        ],
        dtype=np.float32,
    )
    vec_bc = np.array(
        [
            point_c["x"] - point_b["x"],
            point_c["y"] - point_b["y"],
            point_c["z"] - point_b["z"],
        ],
        dtype=np.float32,
    )
    mag_ba = float(np.linalg.norm(vec_ba))
    mag_bc = float(np.linalg.norm(vec_bc))
    if mag_ba < 1e-5 or mag_bc < 1e-5:
        return None

    cos_value = float(np.dot(vec_ba, vec_bc) / (mag_ba * mag_bc))
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def midpoint(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        "x": (a["x"] + b["x"]) / 2.0,
        "y": (a["y"] + b["y"]) / 2.0,
        "z": (a["z"] + b["z"]) / 2.0,
        "likelihood": min(a["likelihood"], b["likelihood"]),
    }


def distance2(a: dict[str, float], b: dict[str, float]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def average_optional(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2.0


def normalize_numeric_stem(stem: str) -> str | None:
    if not stem:
        return None
    if not stem.isdigit():
        return None
    return str(int(stem))


def normalize_exercise(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized == "push-up":
        normalized = "pushup"
    if normalized in APP_EXERCISES:
        return normalized
    return None


def normalize_view_tag(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if not normalized or normalized == "unknown":
        return None
    if "front" in normalized:
        return "front"
    if "side" in normalized or "profile" in normalized:
        return "side"
    if "oblique" in normalized or "45" in normalized:
        return "oblique"
    return None


def cross_validation_fold(subject_tag: str) -> int:
    hash_value = 0
    for char in subject_tag:
        hash_value = ((hash_value * 31) + ord(char)) & 0x7FFFFFFF
    return hash_value % 5


def synthetic_frame_timestamp(sample_index: int, time_sec: float) -> str:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    offset_ms = int(round(time_sec * 1000.0)) + sample_index
    return (base + timedelta(milliseconds=offset_ms)).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            decoded = json.loads(line)
            if isinstance(decoded, dict):
                items.append(decoded)
    return items


def load_existing_source_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    source_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_sample_id = decoded.get("source_sample_id")
            if isinstance(source_sample_id, str) and source_sample_id:
                source_ids.add(source_sample_id)
    return source_ids


def build_output_record(
    *,
    source_record: dict[str, Any],
    observation: FrameObservation,
    resolved_video_path: Path,
    selection_rank: int,
) -> dict[str, Any]:
    source_sample_id = str(source_record.get("sample_id") or f"fitcoach_{selection_rank:08d}")
    raw_subject_tag = str(
        source_record.get("subject_tag") or f"fitcoach_video_{resolved_video_path.stem}"
    )
    if raw_subject_tag.strip().lower() in {"", "fitcoach_unknown", "subject_unknown"}:
        raw_subject_tag = f"fitcoach_video_{resolved_video_path.stem}"
    subject_tag = raw_subject_tag
    subject_tag = subject_tag.strip().lower().replace(" ", "_")
    if not subject_tag:
        subject_tag = f"fitcoach_video_{resolved_video_path.stem}"

    sample_index = int(source_record.get("sample_index") or 0)
    timestamp_iso = synthetic_frame_timestamp(sample_index, observation.time_sec)
    view_tag = normalize_view_tag(source_record.get("view_tag")) or observation.view_tag
    relative_path = resolved_video_path.as_posix()

    return {
        "sample_id": f"{source_sample_id}_f{observation.frame_index:06d}",
        "source_sample_id": source_sample_id,
        "sample_index": sample_index,
        "source_dataset": "QEVD/FIT-COACH+MediaPipe",
        "video_path": source_record.get("video_path"),
        "resolved_video_path": relative_path,
        "split": source_record.get("split", "unknown"),
        "raw_label": source_record.get("raw_label"),
        "raw_label_descriptive": source_record.get("raw_label_descriptive"),
        "exercise_type": source_record.get("exercise_type"),
        "pose_error_types": source_record.get("pose_error_types", []),
        "quality_tag": source_record.get("quality_tag", "standard"),
        "view_tag": view_tag,
        "subject_tag": subject_tag,
        "annotation_protocol": "v1.4_fitcoach_mediapipe33_auto_single",
        "auto_label_source": "MediaPipe_Pose_Python",
        "manual_checked": False,
        "annotator_id": "fitcoach_mediapipe_batch",
        "cross_validation_fold": cross_validation_fold(subject_tag),
        "depth_mode": "monocular",
        "camera_mode": "single",
        "camera_primary": resolved_video_path.name,
        "smpl_shape_code": None,
        "timestamp": timestamp_iso,
        "capture_time_left": timestamp_iso,
        "user_profile": {
            "gender": "unspecified",
            "age": None,
            "heightCm": None,
            "weightKg": None,
            "experienceLevel": "unknown",
        },
        "frame_index": observation.frame_index,
        "frame_time_sec": round(observation.time_sec, 3),
        "frame_width": observation.frame_width,
        "frame_height": observation.frame_height,
        "frame_selection_rank": selection_rank,
        "frame_quality_score": round(observation.quality_score, 6),
        "frame_core_score": round(observation.core_score, 6),
        "frame_reliable_landmark_count": observation.reliable_landmark_count,
        "target_time_sec": source_record.get("target_time_sec"),
        "metric_snapshot": observation.metrics,
        "feedback_candidates": source_record.get("feedback_candidates", []),
        "landmarks_auto": observation.landmarks,
        "landmarks_final": observation.landmarks,
        "landmarks_complete": len(observation.landmarks) == len(LANDMARK_NAMES),
    }


def main() -> None:
    args = parse_args()
    if np is None:
        raise RuntimeError(
            "NumPy is not installed. Please run: pip install -r ml/requirements_extract.txt"
        )
    if sys.version_info >= (3, 13):
        print(
            "[warn] MediaPipe is usually more reliable on Python 3.10/3.11. "
            "If installation fails on 3.13, create a Python 3.11 virtual env for this script."
        )

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    video_root = Path(args.video_root)
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    output_path = Path(args.output)
    report_path = Path(args.report)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    processed_source_ids = load_existing_source_ids(output_path) if args.resume else set()
    allowed_exercises = {
        item.strip().lower()
        for item in args.exercise.split(",")
        if item.strip()
    }

    records = read_jsonl(input_path)
    if args.max_videos > 0:
        records = records[: args.max_videos]

    video_index = VideoIndex(video_root)
    if video_index.video_count < 1000:
        print(
            f"[warn] Only {video_index.video_count} videos were found under {video_root}. "
            "FIT-COACH may not be fully extracted yet."
        )

    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "tqdm is not installed. Please run: pip install -r ml/requirements_extract.txt"
        ) from exc

    extractor = FitCoachLandmarkExtractor(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_visibility=args.min_visibility,
        min_core_score=args.min_core_score,
        min_detected_points=args.min_detected_points,
        rotate_mode=args.rotate,
    )

    counters = Counter()
    resolve_counters = Counter()
    per_exercise_outputs = Counter()
    per_view_outputs = Counter()
    missing_examples: list[str] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for source_record in tqdm(records, desc="FIT-COACH landmark extraction"):
            counters["input_records"] += 1
            exercise = normalize_exercise(source_record.get("exercise_type"))
            if exercise is None:
                counters["unsupported_exercise"] += 1
                continue
            if allowed_exercises and exercise not in allowed_exercises:
                counters["filtered_exercise"] += 1
                continue

            source_sample_id = str(source_record.get("sample_id") or "")
            if source_sample_id and source_sample_id in processed_source_ids:
                counters["skipped_resume"] += 1
                continue

            raw_video_path = str(source_record.get("video_path") or "")
            resolved_video_path, resolve_mode = video_index.resolve(raw_video_path)
            if resolved_video_path is None and args.allow_numeric_fallback:
                resolved_video_path, resolve_mode = video_index.resolve(
                    raw_video_path,
                    allow_numeric_fallback=True,
                )
            resolve_counters[resolve_mode] += 1
            if resolved_video_path is None:
                counters["missing_video"] += 1
                if len(missing_examples) < 20 and raw_video_path:
                    missing_examples.append(raw_video_path)
                continue

            observations = extractor.extract_video(
                video_path=resolved_video_path,
                exercise=exercise,
                probe_frames=args.probe_frames,
                frames_per_video=args.frames_per_video,
                target_time_sec=(
                    float(source_record.get("target_time_sec"))
                    if source_record.get("target_time_sec") is not None
                    else None
                ),
            )
            if not observations:
                counters["no_pose_detected"] += 1
                continue

            counters["videos_with_output"] += 1
            per_exercise_outputs[exercise] += len(observations)
            for selection_rank, observation in enumerate(observations, start=1):
                output_record = build_output_record(
                    source_record=source_record,
                    observation=observation,
                    resolved_video_path=resolved_video_path,
                    selection_rank=selection_rank,
                )
                per_view_outputs[output_record["view_tag"]] += 1
                handle.write(json.dumps(output_record, ensure_ascii=False))
                handle.write("\n")
                counters["output_samples"] += 1

            if source_sample_id:
                processed_source_ids.add(source_sample_id)

    extractor.close()

    report = {
        "generated_at": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "input_jsonl": str(input_path),
        "video_root": str(video_root),
        "output_jsonl": str(output_path),
        "report_path": str(report_path),
        "video_count_under_root": video_index.video_count,
        "frames_per_video": args.frames_per_video,
        "probe_frames": args.probe_frames,
        "min_visibility": args.min_visibility,
        "min_core_score": args.min_core_score,
        "min_detected_points": args.min_detected_points,
        "counts": dict(counters),
        "resolve_modes": dict(resolve_counters),
        "per_exercise_outputs": dict(per_exercise_outputs),
        "per_view_outputs": dict(per_view_outputs),
        "missing_video_examples": missing_examples,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Output JSONL written to: {output_path}")
    print(f"Report written to: {report_path}")
    print(f"Output samples: {counters['output_samples']}")
    print(f"Videos with output: {counters['videos_with_output']}")
    print(f"Missing videos: {counters['missing_video']}")


if __name__ == "__main__":
    main()

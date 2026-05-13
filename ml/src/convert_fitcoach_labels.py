from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# Align to app-side enum ExerciseType
APP_EXERCISES = {"squat", "pushup", "plank"}

# Align to app-side enum PoseErrorType in lib/models/analysis_result.dart
APP_POSE_ERROR_TYPES = {
    "kneeValgus",
    "kneeOverToe",
    "shallowSquat",
    "torsoLeanForward",
    "pushupDepthNotEnough",
    "pushupHipSag",
    "pushupHipPike",
    "pushupElbowFlare",
    "plankHipSag",
    "plankHipPike",
    "plankNeckNotNeutral",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert QEVD-FIT-COACH fine-grained labels into a JSONL format "
            "aligned with the current Flutter app label schema."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to fine_grained_labels.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for mapped JSONL",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional path for conversion report JSON",
    )
    parser.add_argument(
        "--feedback-short",
        type=str,
        default="",
        help="Optional path to feedbacks_short_clips.json",
    )
    parser.add_argument(
        "--feedback-long",
        type=str,
        default="",
        help="Optional path to feedbacks_long_range.json",
    )
    parser.add_argument(
        "--app-only",
        action="store_true",
        help="Only keep records that map to app-supported exercises.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def detect_exercise(label_text: str) -> str | None:
    t = normalize_text(label_text)
    if "squat" in t:
        return "squat"
    if "push up" in t or "pushup" in t:
        return "pushup"
    if "plank" in t:
        return "plank"
    return None


def map_error_types(exercise: str | None, label_text: str) -> list[str]:
    if exercise is None:
        return []

    t = normalize_text(label_text)
    mapped: list[str] = []

    if exercise == "squat":
        if "knee cave" in t or "knee valgus" in t or "knees inward" in t:
            mapped.append("kneeValgus")
        if "too shallow" in t or "depth not enough" in t:
            mapped.append("shallowSquat")
        if "leaning forward" in t or "torso too forward" in t:
            mapped.append("torsoLeanForward")
        if "knee over toe" in t or "knees too far forward" in t:
            mapped.append("kneeOverToe")

    elif exercise == "pushup":
        if "not deep enough" in t or "depth not enough" in t:
            mapped.append("pushupDepthNotEnough")
        if "hips sag" in t or "lower back sag" in t:
            mapped.append("pushupHipSag")
        if "hips too high" in t or "pike" in t:
            mapped.append("pushupHipPike")
        if "elbows too wide" in t or "elbow flare" in t:
            mapped.append("pushupElbowFlare")

    elif exercise == "plank":
        if "hips on the floor" in t or "legs and hips on the floor" in t or "sag" in t:
            mapped.append("plankHipSag")
        if "hips too high" in t or "pike" in t:
            mapped.append("plankHipPike")
        if "neck" in t and ("not neutral" in t or "too high" in t or "too low" in t):
            mapped.append("plankNeckNotNeutral")

    # Keep only valid enum values and de-duplicate
    uniq = []
    seen = set()
    for item in mapped:
        if item in APP_POSE_ERROR_TYPES and item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _load_feedback_index(path: str) -> dict[str, list[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, list):
        return {}

    out: dict[str, list[str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        video_path = str(item.get("video_path", "")).strip()
        feedbacks = item.get("feedbacks", [])
        if not video_path or not isinstance(feedbacks, list):
            continue
        normalized = []
        for fb in feedbacks:
            s = str(fb).strip()
            if s:
                normalized.append(s)
        if normalized:
            out[video_path] = normalized
    return out


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def to_records(
    items: list[dict[str, Any]],
    feedback_index: dict[str, list[str]],
    app_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_counter: Counter[str] = Counter()
    exercise_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()
    unresolved_exercise = 0
    unresolved_error = 0
    multi_label_samples = 0

    for idx, item in enumerate(items):
        video_path = str(item.get("video_path", "")).strip()
        labels = item.get("labels", [])
        labels_desc = item.get("labels_descriptive", [])
        split = str(item.get("split", "unknown")).strip().lower()
        if not isinstance(labels, list):
            labels = []
        if not isinstance(labels_desc, list):
            labels_desc = []

        split_counter[split] += 1
        if len(labels) > 1:
            multi_label_samples += 1

        # Keep first label as canonical, but map using all labels for better recall.
        raw_label = str(labels[0]) if labels else ""
        raw_label_desc = str(labels_desc[0]) if labels_desc else ""

        exercise = None
        for label in labels:
            exercise = detect_exercise(str(label))
            if exercise is not None:
                break
        if exercise is None:
            for label in labels_desc:
                exercise = detect_exercise(str(label))
                if exercise is not None:
                    break
        if exercise is None:
            unresolved_exercise += 1
        else:
            exercise_counter[exercise] += 1

        mapped_errors: list[str] = []
        for label in labels:
            mapped_errors.extend(map_error_types(exercise, str(label)))
        for label in labels_desc:
            mapped_errors.extend(map_error_types(exercise, str(label)))
        mapped_errors = _dedupe_keep_order(mapped_errors)
        if exercise is not None and not mapped_errors:
            unresolved_error += 1
        for err in mapped_errors:
            error_counter[err] += 1

        quality_tag = "error" if mapped_errors else "standard"
        feedback_candidates = feedback_index.get(video_path, [])
        keep = (exercise in APP_EXERCISES) if app_only else True
        if not keep:
            continue

        record = {
            "sample_id": f"fitcoach_{idx:08d}",
            "sample_index": idx,
            "source_dataset": "QEVD/FIT-COACH",
            "video_path": video_path,
            "split": split,
            "raw_label": raw_label,
            "raw_label_descriptive": raw_label_desc,
            "exercise_type": exercise,
            "pose_error_types": mapped_errors,
            "quality_tag": quality_tag,
            "view_tag": "unknown",
            "subject_tag": "fitcoach_unknown",
            "annotator_id": "fitcoach_auto_mapper",
            "manual_checked": False,
            "depth_mode": "monocular",
            "camera_mode": "single",
            "landmarks_complete": False,
            "feedback_candidates": feedback_candidates,
        }
        records.append(record)

    report = {
        "total_samples": len(records),
        "split_counts": dict(split_counter),
        "exercise_counts": dict(exercise_counter),
        "error_counts": dict(error_counter),
        "unresolved_exercise_samples": unresolved_exercise,
        "unresolved_error_samples": unresolved_error,
        "multi_label_samples": multi_label_samples,
        "app_only": app_only,
        "supported_exercises": sorted(APP_EXERCISES),
        "supported_pose_error_types": sorted(APP_POSE_ERROR_TYPES),
    }
    return records, report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of records.")

    feedback_index: dict[str, list[str]] = {}
    short_index = _load_feedback_index(args.feedback_short)
    long_index = _load_feedback_index(args.feedback_long)
    feedback_index.update(long_index)
    feedback_index.update(short_index)

    records, report = to_records(
        raw,
        feedback_index=feedback_index,
        app_only=bool(args.app_only),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Mapped JSONL written to: {output_path}")
    print(f"Report written to: {report_path}")
    print(f"Total samples: {report['total_samples']}")
    print(f"Exercise counts: {report['exercise_counts']}")
    print(f"Unresolved exercise samples: {report['unresolved_exercise_samples']}")
    print(f"Unresolved error samples: {report['unresolved_error_samples']}")


if __name__ == "__main__":
    main()

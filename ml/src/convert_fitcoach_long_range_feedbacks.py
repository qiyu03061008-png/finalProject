from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


APP_EXERCISES = {"squat", "pushup", "plank"}
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
            "Convert FIT-COACH long-range feedback streams into point-in-time "
            "samples for squat, pushup, and plank."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/qevd_fitcoach/extracted/QEVD-FIT-COACH/feedbacks_long_range.json",
        help="Path to feedbacks_long_range.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_long_range_app.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_long_range_app.report.json",
        help="Report JSON path.",
    )
    parser.add_argument(
        "--dedupe-gap-sec",
        type=float,
        default=1.5,
        help="Minimum gap before keeping another sample with the same feedback text.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def relative_seconds(ts: float, base_ts: float) -> float:
    delta = float(ts) - float(base_ts)
    abs_delta = abs(delta)
    # FIT-COACH long-range npy timestamps are stored in nanoseconds.
    # Example frame deltas are around 6.4e7 for ~0.064 seconds.
    if abs_delta > 1e10:
        return delta / 1_000_000_000.0
    if abs_delta > 1e7:
        return delta / 1_000_000.0
    if abs_delta > 1e4:
        return delta / 1_000.0
    return delta


def infer_exercise(text: str) -> str | None:
    t = normalize_text(text)
    if "squat" in t:
        return "squat"
    if "push up" in t or "pushup" in t or "push-ups" in t or "pushups" in t:
        return "pushup"
    if "plank" in t:
        return "plank"
    return None


def is_transition_prompt(text: str) -> bool:
    t = normalize_text(text)
    return (
        t.startswith("first up")
        or t.startswith("moving on to")
        or t.startswith("next up")
        or t.startswith("let's do")
        or t.startswith("lets do")
    )


def map_error_types(exercise: str | None, text: str) -> list[str]:
    if exercise is None:
        return []

    t = normalize_text(text)
    mapped: list[str] = []
    if exercise == "squat":
        if "knees inward" in t or "knee valgus" in t or "knees cave" in t:
            mapped.append("kneeValgus")
        if "shallow" in t or "deeper" in t or "depth" in t:
            mapped.append("shallowSquat")
        if "lean forward" in t or "torso" in t or "chest up" in t:
            mapped.append("torsoLeanForward")
        if "knees too far forward" in t or "knee over toe" in t:
            mapped.append("kneeOverToe")
    elif exercise == "pushup":
        if "not deep enough" in t or "deeper" in t or "shallow pushup" in t:
            mapped.append("pushupDepthNotEnough")
        if "hips sag" in t or "lower back sag" in t or "hips too low" in t:
            mapped.append("pushupHipSag")
        if "hips too high" in t or "pike" in t:
            mapped.append("pushupHipPike")
        if "arms too wide" in t or "elbows too wide" in t or "elbow flare" in t:
            mapped.append("pushupElbowFlare")
    elif exercise == "plank":
        if "hips on the floor" in t or "legs and hips on the floor" in t or "butt sinking" in t:
            mapped.append("plankHipSag")
        if "hips too high" in t or "high hips" in t or "pike" in t:
            mapped.append("plankHipPike")
        if "head up" in t or "head down" in t or "neck" in t:
            mapped.append("plankNeckNotNeutral")

    out: list[str] = []
    seen = set()
    for item in mapped:
        if item in APP_POSE_ERROR_TYPES and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("feedbacks_long_range.json must be a list.")

    records: list[dict[str, Any]] = []
    videos_with_supported_feedbacks = 0
    skipped_transition_prompts = 0
    missing_timestamp_files = 0

    for item in raw:
        if not isinstance(item, dict):
            continue
        long_video_file = str(item.get("long_range_video_file") or "").strip()
        video_timestamps_rel = str(item.get("video_timestamps") or "").strip()
        feedbacks = item.get("feedbacks", [])
        if not long_video_file or not video_timestamps_rel or not isinstance(feedbacks, list):
            continue

        timestamps_path = (input_path.parent / video_timestamps_rel).resolve()
        if not timestamps_path.exists():
            missing_timestamp_files += 1
            continue

        timestamps = np.load(str(timestamps_path))
        usable_len = min(len(timestamps), len(feedbacks))
        timestamps = timestamps[:usable_len]
        feedbacks = feedbacks[:usable_len]
        if usable_len == 0:
            continue

        kept_for_video = 0
        last_kept_by_text: dict[str, float] = {}
        base_ts = float(timestamps[0])
        video_stem = Path(long_video_file).stem

        for frame_idx, (ts, raw_feedback) in enumerate(zip(timestamps, feedbacks)):
            if not isinstance(raw_feedback, str) or not raw_feedback.strip():
                continue
            feedback = raw_feedback.strip()
            exercise = infer_exercise(feedback)
            if exercise not in APP_EXERCISES:
                continue
            if is_transition_prompt(feedback):
                skipped_transition_prompts += 1
                continue

            rel_sec = relative_seconds(float(ts), base_ts)
            norm = normalize_text(feedback)
            last_sec = last_kept_by_text.get(norm)
            if last_sec is not None and abs(rel_sec - last_sec) < args.dedupe_gap_sec:
                continue
            last_kept_by_text[norm] = rel_sec

            pose_error_types = map_error_types(exercise, feedback)
            quality_tag = "error" if pose_error_types else "standard"
            records.append(
                {
                    "sample_id": f"fitcoach_long_{video_stem}_{frame_idx:06d}",
                    "sample_index": len(records),
                    "source_dataset": "QEVD/FIT-COACH-long-range",
                    "video_path": long_video_file,
                    "split": "train",
                    "raw_label": feedback,
                    "raw_label_descriptive": feedback,
                    "exercise_type": exercise,
                    "pose_error_types": pose_error_types,
                    "quality_tag": quality_tag,
                    "view_tag": "unknown",
                    "subject_tag": f"fitcoach_long_{video_stem}",
                    "annotator_id": "fitcoach_long_range_feedbacks",
                    "manual_checked": False,
                    "depth_mode": "monocular",
                    "camera_mode": "single",
                    "landmarks_complete": False,
                    "feedback_candidates": [feedback],
                    "target_time_sec": round(rel_sec, 3),
                }
            )
            kept_for_video += 1

        if kept_for_video > 0:
            videos_with_supported_feedbacks += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    report = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "total_output_records": len(records),
        "videos_with_supported_feedbacks": videos_with_supported_feedbacks,
        "missing_timestamp_files": missing_timestamp_files,
        "skipped_transition_prompts": skipped_transition_prompts,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Long-range JSONL written to: {output_path}")
    print(f"Report written to: {report_path}")
    print(f"Output records: {len(records)}")
    print(f"Videos with supported feedbacks: {videos_with_supported_feedbacks}")


if __name__ == "__main__":
    main()

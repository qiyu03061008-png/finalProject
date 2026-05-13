from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter FIT-COACH mapped JSONL records down to the subset whose videos "
            "can actually be resolved under the extracted video root."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_mapped_app.jsonl",
        help="Input mapped FIT-COACH JSONL.",
    )
    parser.add_argument(
        "--video-root",
        type=str,
        default="data/qevd_fitcoach/extracted/QEVD-FIT-COACH/long_range_videos",
        help="Directory that contains the extracted long-range mp4 files.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_mapped_long_range_app.jsonl",
        help="Filtered JSONL output path.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/qevd_fitcoach/processed/fitcoach_mapped_long_range_app.report.json",
        help="Filter report output path.",
    )
    return parser.parse_args()


def normalize_numeric_stem(stem: str) -> str | None:
    if not stem or not stem.isdigit():
        return None
    return str(int(stem))


def build_available_index(video_root: Path) -> tuple[set[str], set[str]]:
    exact_names: set[str] = set()
    numeric_stems: set[str] = set()
    for path in video_root.rglob("*.mp4"):
        exact_names.add(path.name.lower())
        numeric = normalize_numeric_stem(path.stem)
        if numeric is not None:
            numeric_stems.add(numeric)
    return exact_names, numeric_stems


def can_resolve_video(raw_video_path: str, exact_names: set[str], numeric_stems: set[str]) -> bool:
    raw = raw_video_path.strip()
    if not raw:
        return False
    name = Path(raw).name.lower()
    if name in exact_names:
        return True
    numeric = normalize_numeric_stem(Path(raw).stem)
    if numeric is not None and numeric in numeric_stems:
        return True
    return False


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    video_root = Path(args.video_root)
    output_path = Path(args.output_jsonl)
    report_path = Path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    exact_names, numeric_stems = build_available_index(video_root)
    kept_records: list[dict] = []
    total_records = 0
    missing_records = 0
    missing_examples: list[str] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            raw_video_path = str(record.get("video_path") or "")
            if can_resolve_video(raw_video_path, exact_names, numeric_stems):
                kept_records.append(record)
            else:
                missing_records += 1
                if len(missing_examples) < 20 and raw_video_path:
                    missing_examples.append(raw_video_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in kept_records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    report = {
        "input_jsonl": input_path.as_posix(),
        "video_root": video_root.as_posix(),
        "output_jsonl": output_path.as_posix(),
        "total_records": total_records,
        "kept_records": len(kept_records),
        "missing_records": missing_records,
        "available_exact_video_names": len(exact_names),
        "available_numeric_stems": len(numeric_stems),
        "missing_examples": missing_examples,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Filtered JSONL written to: {output_path}")
    print(f"Report written to: {report_path}")
    print(f"Total records: {total_records}")
    print(f"Kept records: {len(kept_records)}")
    print(f"Missing records: {missing_records}")


if __name__ == "__main__":
    main()

import argparse
import asyncio
import json
import sys
from pathlib import Path


async def _synthesize(edge_tts, text: str, voice: str, output_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cue audio files from assets/audio/cue_manifest.json",
    )
    parser.add_argument(
        "--voice",
        default="zh-CN-XiaoxiaoNeural",
        help="edge-tts voice name",
    )
    parser.add_argument(
        "--manifest",
        default="assets/audio/cue_manifest.json",
        help="Path to cue manifest JSON",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/audio",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--format",
        choices=("mp3",),
        default="mp3",
        help="Output format supported by edge-tts",
    )
    args = parser.parse_args()

    try:
      import edge_tts  # type: ignore
    except ImportError:
        print(
            "edge-tts is not installed. Install it with:\n"
            "  python -m pip install edge-tts",
            file=sys.stderr,
        )
        return 1

    manifest_path = Path(args.manifest)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in entries:
        key = entry["key"]
        text = entry["text"]
        output_path = output_dir / f"{key}.{args.format}"
        print(f"Generating {output_path.name}: {text}")
        asyncio.run(_synthesize(edge_tts, text, args.voice, output_path))

    print(f"Generated {len(entries)} files in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

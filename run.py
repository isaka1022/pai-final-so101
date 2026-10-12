"""Run a language-instructed pick-and-place and record a video: see README."""

import argparse
import json
from pathlib import Path

from pai_final.task import run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instruction", help="instruction sent to the VLM (needs GOOGLE_API_KEY)")
    source.add_argument("--replay", type=Path, help="JSON saved by an earlier run; the VLM is not called")
    parser.add_argument("--out", type=Path, help="output mp4 (default: out/run.mp4, or out/replayed.mp4 with --replay)")
    args = parser.parse_args()

    if args.replay:
        saved = json.loads(args.replay.read_text())
        instruction, raw_response = saved["instruction"], saved["raw_response"]
        out = args.out or Path("out/replayed.mp4")
    else:
        instruction, raw_response = args.instruction, None
        out = args.out or Path("out/run.mp4")

    record = run_episode(instruction, out, raw_response)
    result = record["result"]
    print(f"{result['cube']} -> {result['place']}: success={result['success']} error={result['error_m'] * 1000:.1f} mm steps={result['steps']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

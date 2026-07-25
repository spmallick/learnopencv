"""Run sparse or dense optical flow with OpenCV 4.14 and OpenCV 5."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from algorithms.dense_optical_flow import run_dense_optical_flow
from algorithms.lucas_kanade import run_lucas_kanade


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_DIR / "videos" / "people.mp4"
ALGORITHMS = ("farneback", "lucaskanade", "lucaskanade_dense", "rlof")


def parse_args() -> argparse.Namespace:
    """Parse controls shared by interactive and automated executions."""

    parser = argparse.ArgumentParser(
        description="Visualize sparse and dense optical flow with OpenCV."
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument(
        "--video",
        "--video_path",
        dest="video_path",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Input video (default: {DEFAULT_VIDEO.name}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for the final visualization frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frame pairs; zero processes the whole video.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Dispatch to the requested algorithm and report a stable summary."""

    args = parse_args()
    try:
        if args.max_frames < 0:
            raise ValueError("max-frames cannot be negative")

        common_arguments = {
            "video_path": args.video_path.resolve(),
            "output_dir": args.output_dir.resolve(),
            "show_windows": not args.no_display,
            "max_frames": args.max_frames,
            "validate": args.validate,
        }
        if args.algorithm == "lucaskanade":
            summary = run_lucas_kanade(**common_arguments)
        else:
            summary = run_dense_optical_flow(
                algorithm=args.algorithm,
                **common_arguments,
            )

        print(f"OpenCV version: {cv2.__version__}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Frame pairs processed: {summary.frame_pairs}")
        print(f"Mean motion magnitude: {summary.mean_magnitude:.6f}")
        print(f"Visualization: {summary.output_path}")
        if args.validate:
            print("VALIDATION PASSED: video, motion, and output checks")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

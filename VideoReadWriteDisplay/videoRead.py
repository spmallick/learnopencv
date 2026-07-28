#!/usr/bin/env python3
"""Read a video safely, optionally display it, and report measured metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def read_video(
    input_path: Path,
    *,
    display: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    frames_read = 0
    first_frame_checksum = None
    try:
        while max_frames is None or frames_read < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames_read += 1
            if first_frame_checksum is None:
                first_frame_checksum = int(frame.astype("uint64").sum())
            if display:
                cv2.imshow("Video", frame)
                if cv2.waitKey(25) & 0xFF in (27, ord("q")):
                    break

        if frames_read == 0:
            raise RuntimeError(f"No decodable frames found in: {input_path}")
        return {
            "frames_read": frames_read,
            "frame_size": [
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ],
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "reported_frame_count": int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            ),
            "first_frame_checksum": first_frame_checksum,
        }
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "chaplin.mp4",
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = read_video(
            args.input,
            display=args.display,
            max_frames=args.max_frames,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

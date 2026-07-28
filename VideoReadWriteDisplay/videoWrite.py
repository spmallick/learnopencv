#!/usr/bin/env python3
"""Transcode a video with checked OpenCV VideoCapture and VideoWriter objects."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2


def transcode_video(
    input_path: Path,
    output_path: Path,
    *,
    codec: str = "MJPG",
    output_fps: float | None = None,
    display: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")
    if output_fps is not None and (
        not math.isfinite(output_fps) or output_fps <= 0
    ):
        raise ValueError("output_fps must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    writer: cv2.VideoWriter | None = None
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise RuntimeError(f"No decodable frames found in: {input_path}")

        height, width = first_frame.shape[:2]
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        fps = output_fps or (source_fps if source_fps > 0 else 30.0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open output video '{output_path}' with codec {codec}"
            )

        frames_written = 0
        frame = first_frame
        while max_frames is None or frames_written < max_frames:
            writer.write(frame)
            frames_written += 1
            if display:
                cv2.imshow("Transcoding", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            ok, frame = capture.read()
            if not ok or frame is None:
                break

        return {
            "frames_written": frames_written,
            "frame_size": [width, height],
            "source_fps": source_fps,
            "output_fps": fps,
            "codec": codec,
            "output": str(output_path),
        }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "chaplin.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.avi"),
    )
    parser.add_argument("--codec", default="MJPG")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = transcode_video(
            args.input,
            args.output,
            codec=args.codec,
            output_fps=args.fps,
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

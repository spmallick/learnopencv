#!/usr/bin/env python3
"""Estimate one person's MediaPipe Pose landmarks throughout a video."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from pose_estimation import (
    DEFAULT_MODEL,
    PROJECT_DIR,
    draw_pose,
    infer_pose,
    load_pose_model,
    validate_pose,
)


DEFAULT_INPUT = PROJECT_DIR / "sample_video.mp4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose deterministic video and headless test controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--video_file",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input video (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output",
        help="Directory for pose-video.avi",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; zero processes the complete video.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", action="store_true")
    display_group.add_argument(
        "--no-display",
        action="store_true",
        help="Run headlessly (the default; accepted explicitly for CI).",
    )
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def _validate_written_video(
    output_path: Path,
    expected_size: tuple[int, int],
    expected_frames: int,
) -> None:
    """Reopen the completed video and verify its dimensions and frame count."""

    capture = cv2.VideoCapture(str(output_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen output video: {output_path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if (width, height) != expected_size:
        raise RuntimeError(
            f"Output size {(width, height)} does not match {expected_size}."
        )
    if frame_count != expected_frames:
        raise RuntimeError(
            f"Output frame count {frame_count} does not match {expected_frames}."
        )


def run(args: argparse.Namespace) -> dict[str, object]:
    """Process a video without discarding the first frame or reading past EOF."""

    input_path = args.input.expanduser().resolve()
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Input video has invalid dimensions: {input_path}")
    if not fps or fps <= 0.0:
        fps = 25.0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pose-video.avi"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    net = load_pose_model(args.model, args.device)
    processed = 0
    total_visible = 0
    total_edges = 0
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Decoded an empty frame at index {processed}.")
            if frame.shape[1] != width or frame.shape[0] != height:
                raise RuntimeError(
                    f"Frame {processed} changed size from {(width, height)} "
                    f"to {(frame.shape[1], frame.shape[0])}."
                )

            result = infer_pose(net, frame)
            output, visible_count, edge_count = draw_pose(
                frame, result, args.score_threshold
            )
            if args.validate:
                validate_pose(frame, result, visible_count, edge_count)
            writer.write(output)
            processed += 1
            total_visible += visible_count
            total_edges += edge_count

            if args.display and not args.no_display:
                cv2.imshow("MediaPipe Pose", output)
                if cv2.waitKey(1) == 27:
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if processed == 0:
        raise RuntimeError(f"No frames were decoded from input video: {input_path}")
    if args.validate:
        _validate_written_video(output_path, (width, height), processed)
        print(
            "VALIDATION PASSED: "
            f"frames={processed} size={width}x{height}"
        )

    print(f"OpenCV version: {cv2.__version__}")
    print(
        "POSE VIDEO RESULT: "
        f"frames={processed} total_visible={total_visible} "
        f"total_edges={total_edges}"
    )
    print(f"Saved output: {output_path}")
    return {
        "output": output_path,
        "frames": processed,
        "size": (width, height),
        "total_visible": total_visible,
        "total_edges": total_edges,
    }


def main(argv: list[str] | None = None) -> int:
    """Return a clear nonzero status for missing inputs, models, or writers."""

    try:
        run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Colorize a video with OpenCV DNN and an ONNX model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2 as cv

from colorization import DEFAULT_MODEL, colorize_frame, load_network, validate_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colorize a video frame by frame with OpenCV DNN."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "greyscaleVideo.mp4",
        help="Input video (default: the bundled sample).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to colorization_eccv16.onnx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("colorized-video.avi"),
        help="Destination video.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; 0 processes the entire video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV preview window.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each generated frame and the final frame count.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = cv.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {args.input}")

    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv.CAP_PROP_FPS)
    if not fps or not 1.0 <= fps <= 240.0:
        fps = 30.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv.VideoWriter(
        str(args.output),
        cv.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {args.output}")

    network = load_network(args.model)
    processed = 0
    inference_seconds = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            start = time.perf_counter()
            output, chroma_score = colorize_frame(frame, network)
            inference_seconds += time.perf_counter() - start
            if args.validate:
                validate_output(frame, output, chroma_score)

            writer.write(output)
            processed += 1
            if not args.no_display:
                cv.imshow("Colorized video", output)
                if cv.waitKey(1) & 0xFF == 27:
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        capture.release()
        writer.release()
        if not args.no_display:
            cv.destroyAllWindows()

    if args.validate and processed == 0:
        raise RuntimeError("No frames were processed.")

    average = inference_seconds / processed if processed else 0.0
    print(f"Saved {processed} frames to {args.output}")
    print(f"Average inference time: {average:.3f} seconds per frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

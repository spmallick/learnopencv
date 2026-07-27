#!/usr/bin/env python3
"""Detect faces in an image or video with OpenCV FaceDetectorYN and YuNet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from face_detection import (
    DEFAULT_YUNET_MODEL,
    PROJECT_DIR,
    YuNetDetector,
    draw_detections,
    validate_detections,
    write_image,
)


DEFAULT_INPUT = PROJECT_DIR / "videos" / "baby.mp4"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose reproducible image/video, output, device, and validation controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--video",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image or video (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "image", "video"),
        default="auto",
        help="Input type; auto uses the filename extension.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_YUNET_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Video only: zero processes the complete input.",
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


def infer_mode(input_path: Path, requested_mode: str) -> str:
    """Resolve auto mode without attempting to decode the same video twice."""

    if requested_mode != "auto":
        return requested_mode
    return "image" if input_path.suffix.lower() in IMAGE_SUFFIXES else "video"


def _validate_written_video(
    output_path: Path,
    expected_size: tuple[int, int],
    expected_frames: int,
) -> None:
    """Reopen the saved AVI to verify dimensions and exact requested frame count."""

    capture = cv2.VideoCapture(str(output_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen output video: {output_path}")
    actual_size = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if actual_size != expected_size:
        raise RuntimeError(
            f"Output size {actual_size} does not match {expected_size}."
        )
    if actual_frames != expected_frames:
        raise RuntimeError(
            f"Output frame count {actual_frames} does not match {expected_frames}."
        )


def run_image(
    args: argparse.Namespace,
    detector: YuNetDetector,
    input_path: Path,
) -> dict[str, object]:
    """Run one image through YuNet and save a same-size annotated JPEG."""

    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    detections = detector.detect(frame)
    if args.validate:
        validate_detections(frame, detections)
    output = draw_detections(frame, detections, detector.name)
    output_path = args.output_dir.expanduser().resolve() / "yunet-image.jpg"
    write_image(output_path, output)

    if args.validate:
        saved = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if saved is None or saved.shape != frame.shape:
            raise RuntimeError("Saved image is unreadable or changed dimensions.")
        print(f"VALIDATION PASSED: mode=image faces={len(detections)}")
    if args.display and not args.no_display:
        cv2.imshow("YuNet Face Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"FACE RESULT: detector=YuNet faces={len(detections)}")
    print(f"Saved output: {output_path}")
    return {
        "output": output_path,
        "faces": len(detections),
        "shape": output.shape,
    }


def run_video(
    args: argparse.Namespace,
    detector: YuNetDetector,
    input_path: Path,
) -> dict[str, object]:
    """Process every frame exactly once and preserve source size and frame rate."""

    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid dimensions.")
    if not fps or fps <= 0.0:
        fps = 25.0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "yunet-video.avi"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    processed = 0
    total_faces = 0
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Decoded an empty frame at index {processed}.")
            if (frame.shape[1], frame.shape[0]) != (width, height):
                raise RuntimeError("A decoded frame changed dimensions.")

            detections = detector.detect(frame)
            if args.validate:
                validate_detections(frame, detections)
            output = draw_detections(frame, detections, detector.name)
            writer.write(output)
            processed += 1
            total_faces += len(detections)

            if args.display and not args.no_display:
                cv2.imshow("YuNet Face Detection", output)
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
            f"mode=video frames={processed} size={width}x{height}"
        )

    print(
        "FACE VIDEO RESULT: "
        f"detector=YuNet frames={processed} total_faces={total_faces}"
    )
    print(f"Saved output: {output_path}")
    return {
        "output": output_path,
        "frames": processed,
        "size": (width, height),
        "total_faces": total_faces,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Construct the detector once, then dispatch to image or video handling."""

    input_path = args.input.expanduser().resolve()
    detector = YuNetDetector(
        args.model,
        args.score_threshold,
        args.nms_threshold,
        args.top_k,
        args.device,
    )
    mode = infer_mode(input_path, args.mode)
    result = (
        run_image(args, detector, input_path)
        if mode == "image"
        else run_video(args, detector, input_path)
    )
    print(f"OpenCV version: {cv2.__version__}")
    return result


def main(argv: list[str] | None = None) -> int:
    """Return a clear nonzero status for input, model, writer, and DNN errors."""

    try:
        run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

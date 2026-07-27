#!/usr/bin/env python3
"""Compare YuNet with optional Haar and dlib HOG face-detector baselines."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from face_detection import (
    DEFAULT_YUNET_MODEL,
    PROJECT_DIR,
    Detector,
    create_detector,
    draw_detections,
    validate_detections,
    write_image,
)
from face_detection_opencv_dnn import infer_mode


DEFAULT_INPUT = PROJECT_DIR / "videos" / "baby.mp4"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a comparison list while keeping YuNet-only defaults dependency-free."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--video",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument("--mode", choices=("auto", "image", "video"), default="auto")
    parser.add_argument("--model", type=Path, default=DEFAULT_YUNET_MODEL)
    parser.add_argument(
        "--detectors",
        default="yunet",
        help="Comma-separated subset of yunet,haar,hog (default: yunet).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output-comparison",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=5000)
    parser.add_argument("--max-frames", type=int, default=0)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", action="store_true")
    display_group.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def build_detectors(args: argparse.Namespace) -> list[Detector]:
    """Instantiate only explicitly requested optional dependencies."""

    names = [name.strip() for name in args.detectors.split(",") if name.strip()]
    if not names:
        raise ValueError("--detectors must name at least one detector.")
    normalized = [name.lower() for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("--detectors cannot contain duplicates.")
    return [
        create_detector(
            name,
            model_path=args.model,
            score_threshold=args.score_threshold,
            nms_threshold=args.nms_threshold,
            top_k=args.top_k,
            device=args.device,
        )
        for name in names
    ]


def compare_frame(
    frame: np.ndarray,
    detectors: list[Detector],
    validate: bool,
) -> tuple[np.ndarray, list[int]]:
    """Run each detector on the same pixels and join same-size panels horizontally."""

    panels: list[np.ndarray] = []
    counts: list[int] = []
    for detector in detectors:
        detections = detector.detect(frame)
        if validate:
            validate_detections(frame, detections)
        panels.append(draw_detections(frame, detections, detector.name))
        counts.append(len(detections))
    comparison = cv2.hconcat(panels)
    expected_shape = (frame.shape[0], frame.shape[1] * len(detectors), 3)
    if comparison.shape != expected_shape:
        raise RuntimeError(
            f"Comparison shape {comparison.shape} does not match {expected_shape}."
        )
    return comparison, counts


def _validate_written_video(
    output_path: Path,
    expected_size: tuple[int, int],
    expected_frames: int,
) -> None:
    """Reopen the comparison video and verify writer geometry and frame count."""

    capture = cv2.VideoCapture(str(output_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen output video: {output_path}")
    actual_size = (
        int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    actual_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    if actual_size != expected_size or actual_frames != expected_frames:
        raise RuntimeError(
            "Saved comparison video has unexpected dimensions or frame count."
        )


def run_image(
    args: argparse.Namespace,
    detectors: list[Detector],
    input_path: Path,
) -> dict[str, object]:
    """Save one horizontal comparison image."""

    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    comparison, counts = compare_frame(frame, detectors, args.validate)
    output_path = args.output_dir.expanduser().resolve() / "comparison-image.jpg"
    write_image(output_path, comparison)

    if args.validate:
        saved = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if saved is None or saved.shape != comparison.shape:
            raise RuntimeError("Saved comparison image is unreadable or resized.")
        print(
            "VALIDATION PASSED: "
            f"mode=image panels={len(detectors)} size="
            f"{comparison.shape[1]}x{comparison.shape[0]}"
        )
    if args.display and not args.no_display:
        cv2.imshow("Face Detection Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(
        "COMPARISON RESULT: "
        + " ".join(
            f"{detector.name}={count}"
            for detector, count in zip(detectors, counts)
        )
    )
    print(f"Saved output: {output_path}")
    return {"output": output_path, "counts": counts, "shape": comparison.shape}


def run_video(
    args: argparse.Namespace,
    detectors: list[Detector],
    input_path: Path,
) -> dict[str, object]:
    """Save a correctly sized horizontal comparison video."""

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
    output_path = output_dir / "comparison-video.avi"
    output_size = (width * len(detectors), height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    processed = 0
    totals = [0 for _ in detectors]
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Decoded an empty frame at index {processed}.")
            comparison, counts = compare_frame(frame, detectors, args.validate)
            writer.write(comparison)
            processed += 1
            totals = [total + count for total, count in zip(totals, counts)]

            if args.display and not args.no_display:
                cv2.imshow("Face Detection Comparison", comparison)
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
        _validate_written_video(output_path, output_size, processed)
        print(
            "VALIDATION PASSED: "
            f"mode=video panels={len(detectors)} frames={processed} "
            f"size={output_size[0]}x{output_size[1]}"
        )
    print(
        "COMPARISON VIDEO RESULT: "
        f"frames={processed} "
        + " ".join(
            f"{detector.name}={total}"
            for detector, total in zip(detectors, totals)
        )
    )
    print(f"Saved output: {output_path}")
    return {
        "output": output_path,
        "frames": processed,
        "totals": totals,
        "size": output_size,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Build selected detectors and dispatch based on the input type."""

    input_path = args.input.expanduser().resolve()
    detectors = build_detectors(args)
    mode = infer_mode(input_path, args.mode)
    result = (
        run_image(args, detectors, input_path)
        if mode == "image"
        else run_video(args, detectors, input_path)
    )
    print(f"OpenCV version: {cv2.__version__}")
    return result


def main(argv: list[str] | None = None) -> int:
    """Convert dependency, input, output, and detector errors to exit code 2."""

    try:
        run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

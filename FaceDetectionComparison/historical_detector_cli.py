"""Reusable image/video runner for optional historical detector baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from face_detection import Detector, PROJECT_DIR, draw_detections, validate_detections, write_image
from face_detection_opencv_dnn import infer_mode


DEFAULT_INPUT = PROJECT_DIR / "videos" / "baby.mp4"


def parse_historical_args(
    description: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Give historical examples the same safe I/O controls as the YuNet path."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        "--video",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument("--mode", choices=("auto", "image", "video"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output-historical",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--display", action="store_true")
    display_group.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def run_historical(
    args: argparse.Namespace,
    detector: Detector,
    output_stem: str,
) -> dict[str, object]:
    """Run a selected baseline without introducing it into default tests."""

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    mode = infer_mode(input_path, args.mode)
    if mode == "image":
        frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise FileNotFoundError(f"Could not read input image: {input_path}")
        detections = detector.detect(frame)
        if args.validate:
            validate_detections(frame, detections)
        output = draw_detections(frame, detections, detector.name)
        output_path = output_dir / f"{output_stem}-image.jpg"
        write_image(output_path, output)
        if args.validate:
            print(
                f"VALIDATION PASSED: mode=image faces={len(detections)} "
                f"detector={detector.name}"
            )
        if args.display and not args.no_display:
            cv2.imshow(detector.name, output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Saved output: {output_path}")
        return {"output": output_path, "faces": len(detections)}

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

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_stem}-video.avi"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frames = 0
    total_faces = 0
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            detections = detector.detect(frame)
            if args.validate:
                validate_detections(frame, detections)
            output = draw_detections(frame, detections, detector.name)
            writer.write(output)
            frames += 1
            total_faces += len(detections)
            if args.display and not args.no_display:
                cv2.imshow(detector.name, output)
                if cv2.waitKey(1) == 27:
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if frames == 0:
        raise RuntimeError(f"No frames were decoded from input video: {input_path}")
    if args.validate:
        check = cv2.VideoCapture(str(output_path))
        actual_size = (
            int(round(check.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(check.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        actual_frames = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT)))
        check.release()
        if actual_size != (width, height) or actual_frames != frames:
            raise RuntimeError("Saved historical video failed output validation.")
        print(
            f"VALIDATION PASSED: mode=video frames={frames} "
            f"size={width}x{height} detector={detector.name}"
        )
    print(f"Saved output: {output_path}")
    return {
        "output": output_path,
        "frames": frames,
        "total_faces": total_faces,
    }

#!/usr/bin/env python3
"""Estimate one person's MediaPipe Pose landmarks in an image."""

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
    write_image,
)


DEFAULT_INPUT = PROJECT_DIR / "single.jpeg"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose reproducible input, output, device, display, and validation controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--image_file",
        dest="input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"MediaPipe Pose ONNX model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "output",
        help="Directory for pose-image.jpg",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="DNN execution device; CUDA requires a CUDA-enabled OpenCV build.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum landmark visibility and presence probability.",
    )
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        action="store_true",
        help="Open an interactive result window.",
    )
    display_group.add_argument(
        "--no-display",
        action="store_true",
        help="Run headlessly (the default; accepted explicitly for CI).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check stable output invariants and print a success marker.",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the real image pipeline and return metrics used by regression tests."""

    input_path = args.input.expanduser().resolve()
    frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise FileNotFoundError(f"Could not read input image: {input_path}")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1.")

    net = load_pose_model(args.model, args.device)
    result = infer_pose(net, frame)
    output, visible_count, edge_count = draw_pose(
        frame, result, args.score_threshold
    )
    output_path = args.output_dir.expanduser().resolve() / "pose-image.jpg"
    write_image(output_path, output)

    if args.validate:
        validate_pose(frame, result, visible_count, edge_count)
        print(
            "VALIDATION PASSED: "
            f"landmarks=33 visible={visible_count} edges={edge_count}"
        )

    if args.display and not args.no_display:
        cv2.imshow("MediaPipe Pose", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"OpenCV version: {cv2.__version__}")
    print(
        "POSE RESULT: "
        f"confidence={result.confidence:.6f} "
        f"visible={visible_count} edges={edge_count}"
    )
    print(f"Saved output: {output_path}")
    return {
        "output": output_path,
        "shape": output.shape,
        "confidence": result.confidence,
        "visible": visible_count,
        "edges": edge_count,
    }


def main(argv: list[str] | None = None) -> int:
    """Translate expected user errors into a concise nonzero CLI exit."""

    try:
        run(parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Detect circles with OpenCV's gradient Hough transform."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from hough_utils import (
    circle_output_paths,
    detect_circles,
    draw_circles,
    read_bgr,
    resolve_input,
    write_image,
)


def run_validation(output_dir: str | Path | None = None) -> dict[str, int]:
    """Validate a known synthetic circle plus the bundled eye image."""

    synthetic = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(synthetic, (128, 128), 60, (255, 255, 255), 3, cv2.LINE_AA)
    _, synthetic_circles = detect_circles(
        synthetic,
        dp=1.2,
        min_distance=40,
        param1=120,
        param2=30,
        min_radius=50,
        max_radius=70,
    )
    if synthetic_circles.size == 0:
        raise AssertionError("synthetic circle was not detected")
    distances = np.hypot(
        synthetic_circles[:, 0] - 128.0,
        synthetic_circles[:, 1] - 128.0,
    )
    radius_errors = np.abs(synthetic_circles[:, 2] - 60.0)
    if not np.any((distances <= 4.0) & (radius_errors <= 4.0)):
        raise AssertionError(f"synthetic circle geometry was inaccurate: {synthetic_circles}")

    input_path = resolve_input(None, "brown-eyes.jpg")
    image = read_bgr(input_path)
    blurred, circles = detect_circles(
        image,
        dp=1.2,
        min_distance=image.shape[0] / 4.0,
        param1=120,
        param2=30,
        min_radius=25,
        max_radius=55,
    )
    if circles.size == 0:
        raise AssertionError("no circles were detected in brown-eyes.jpg")
    annotated = draw_circles(image, circles)

    if output_dir is not None:
        blur_path, result_path = circle_output_paths(output_dir, input_path.stem)
        write_image(blur_path, blurred)
        write_image(result_path, annotated)

    result = {
        "circle_count": int(circles.shape[0]),
        "synthetic_circle_count": int(synthetic_circles.shape[0]),
    }
    print(
        "VALIDATION PASSED: "
        f"circles={result['circle_count']}, "
        f"synthetic_circles={result['synthetic_circle_count']}"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Create the circle-detector CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, help="input image; defaults to brown-eyes.jpg"
    )
    parser.add_argument("--output-dir", type=Path, help="directory for blurred/result images")
    parser.add_argument("--dp", type=float, default=1.2)
    parser.add_argument("--min-distance", type=float)
    parser.add_argument("--param1", type=float, default=120.0)
    parser.add_argument("--param2", type=float, default=30.0)
    parser.add_argument("--min-radius", type=int, default=25)
    parser.add_argument("--max-radius", type=int, default=55)
    parser.add_argument("--no-display", action="store_true", help="do not open GUI windows")
    parser.add_argument("--validate", action="store_true", help="run deterministic checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Detect circles, report metrics, and optionally save/display outputs."""

    arguments = _build_parser().parse_args(argv)
    if arguments.validate:
        run_validation(arguments.output_dir)
        return 0

    input_path = resolve_input(arguments.input, "brown-eyes.jpg")
    image = read_bgr(input_path)
    min_distance = arguments.min_distance or image.shape[0] / 4.0
    blurred, circles = detect_circles(
        image,
        dp=arguments.dp,
        min_distance=min_distance,
        param1=arguments.param1,
        param2=arguments.param2,
        min_radius=arguments.min_radius,
        max_radius=arguments.max_radius,
    )
    annotated = draw_circles(image, circles)
    print(f"Input: {input_path}")
    print(f"Circles: {circles.shape[0]}")

    if arguments.output_dir is not None:
        blur_path, result_path = circle_output_paths(
            arguments.output_dir, input_path.stem
        )
        write_image(blur_path, blurred)
        write_image(result_path, annotated)
        print(f"Wrote: {blur_path}")
        print(f"Wrote: {result_path}")

    if not arguments.no_display:
        cv2.imshow("Median-blurred grayscale", blurred)
        cv2.imshow("Hough circles", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

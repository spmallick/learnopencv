#!/usr/bin/env python3
"""Detect line segments with Canny edges and the probabilistic Hough transform."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from hough_utils import (
    detect_lines,
    draw_lines,
    line_output_paths,
    read_bgr,
    resolve_input,
    write_image,
)


def _has_horizontal_line(lines: np.ndarray, minimum_length: int = 150) -> bool:
    """Return True when a long, nearly horizontal semantic line is present."""

    for x1, y1, x2, y2 in lines:
        if abs(int(y2) - int(y1)) <= 3 and abs(int(x2) - int(x1)) >= minimum_length:
            return True
    return False


def _has_diagonal_line(lines: np.ndarray, minimum_length: float = 130.0) -> bool:
    """Return True when a long diagonal semantic line is present."""

    for x1, y1, x2, y2 in lines:
        delta_x = int(x2) - int(x1)
        delta_y = int(y2) - int(y1)
        length = float(np.hypot(delta_x, delta_y))
        if length >= minimum_length and abs(delta_x) > 20 and abs(delta_y) > 20:
            return True
    return False


def run_validation(output_dir: str | Path | None = None) -> dict[str, int]:
    """Validate semantic lines on a synthetic image plus the bundled road image."""

    synthetic = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.line(synthetic, (20, 45), (235, 45), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(synthetic, (25, 225), (225, 75), (255, 255, 255), 3, cv2.LINE_AA)
    _, synthetic_lines = detect_lines(
        synthetic,
        hough_threshold=45,
        min_line_length=100,
        max_line_gap=12,
    )
    if not _has_horizontal_line(synthetic_lines):
        raise AssertionError("synthetic horizontal line was not recovered")
    if not _has_diagonal_line(synthetic_lines):
        raise AssertionError("synthetic diagonal line was not recovered")

    input_path = resolve_input(None, "lanes.jpg")
    image = read_bgr(input_path)
    edges, lines = detect_lines(image)
    if lines.size == 0:
        raise AssertionError("no lines were detected in lanes.jpg")
    annotated = draw_lines(image, lines)

    if output_dir is not None:
        edge_path, result_path = line_output_paths(output_dir, input_path.stem)
        write_image(edge_path, edges)
        write_image(result_path, annotated)

    result = {
        "line_count": int(lines.shape[0]),
        "edge_pixels": int(cv2.countNonZero(edges)),
        "synthetic_line_count": int(synthetic_lines.shape[0]),
    }
    print(
        "VALIDATION PASSED: "
        f"lines={result['line_count']}, "
        f"edge_pixels={result['edge_pixels']}, "
        f"synthetic_lines={result['synthetic_line_count']}"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Create the line-detector CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="input image; defaults to lanes.jpg")
    parser.add_argument("--output-dir", type=Path, help="directory for edges and line result")
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--hough-threshold", type=int, default=50)
    parser.add_argument("--min-line-length", type=int, default=40)
    parser.add_argument("--max-line-gap", type=int, default=25)
    parser.add_argument("--no-display", action="store_true", help="do not open GUI windows")
    parser.add_argument("--validate", action="store_true", help="run deterministic checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Detect lines, report metrics, and optionally save/display outputs."""

    arguments = _build_parser().parse_args(argv)
    if arguments.validate:
        run_validation(arguments.output_dir)
        return 0

    input_path = resolve_input(arguments.input, "lanes.jpg")
    image = read_bgr(input_path)
    edges, lines = detect_lines(
        image,
        canny_low=arguments.canny_low,
        canny_high=arguments.canny_high,
        hough_threshold=arguments.hough_threshold,
        min_line_length=arguments.min_line_length,
        max_line_gap=arguments.max_line_gap,
    )
    annotated = draw_lines(image, lines)
    print(f"Input: {input_path}")
    print(f"Line segments: {lines.shape[0]}")
    print(f"Edge pixels: {cv2.countNonZero(edges)}")

    if arguments.output_dir is not None:
        edge_path, result_path = line_output_paths(
            arguments.output_dir, input_path.stem
        )
        write_image(edge_path, edges)
        write_image(result_path, annotated)
        print(f"Wrote: {edge_path}")
        print(f"Wrote: {result_path}")

    if not arguments.no_display:
        cv2.imshow("Edges", edges)
        cv2.imshow("Probabilistic Hough lines", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

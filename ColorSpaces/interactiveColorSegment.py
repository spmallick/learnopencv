#!/usr/bin/env python3
"""Segment a BGR image with thresholds defined in a chosen color space."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from color_spaces import (
    COLOR_CONVERSIONS,
    apply_mask,
    read_bgr,
    resolve_input_path,
    run_core_validation,
    threshold_mask,
    write_image,
)


def segment_image(
    image: np.ndarray,
    color_space: str,
    lower: Sequence[int],
    upper: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-channel mask and a BGR visualization."""

    mask = threshold_mask(image, color_space, lower, upper)
    return mask, apply_mask(image, mask)


def output_paths(output_dir: str | Path, stem: str, color_space: str) -> tuple[Path, Path]:
    """Return deterministic mask and visualization output paths."""

    directory = Path(output_dir).expanduser().resolve()
    normalized_space = color_space.lower()
    return (
        directory / f"{stem}-{normalized_space}-mask.png",
        directory / f"{stem}-{normalized_space}-result.png",
    )


def run_validation(output_dir: str | Path | None = None) -> dict[str, int]:
    """Validate the default yellow segmentation on a bundled image."""

    metrics = run_core_validation()
    input_path = resolve_input_path(None)
    image = read_bgr(input_path)
    mask, result = segment_image(image, "HSV", (20, 80, 40), (45, 255, 255))
    foreground_pixels = int(cv2.countNonZero(mask))
    if foreground_pixels <= 0 or foreground_pixels >= mask.size:
        raise AssertionError(f"unexpected foreground pixel count: {foreground_pixels}")
    if result.shape != image.shape:
        raise AssertionError("segmentation result shape differs from input")

    if output_dir is not None:
        mask_path, result_path = output_paths(output_dir, input_path.stem, "HSV")
        write_image(mask_path, mask)
        write_image(result_path, result)

    print(
        "VALIDATION PASSED: "
        f"foreground_pixels={foreground_pixels}, "
        f"image={image.shape[1]}x{image.shape[0]}"
    )
    return {**metrics, "foreground_pixels": foreground_pixels}


def _triplet(values: list[str]) -> tuple[int, int, int]:
    """Parse three integer threshold values for argparse."""

    try:
        parsed = tuple(int(value) for value in values)
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be integers") from error
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("thresholds require three values")
    return parsed  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="input image; defaults to images/rub00.jpg")
    parser.add_argument(
        "--space",
        choices=tuple(COLOR_CONVERSIONS),
        default="HSV",
        help="color space in which thresholds are interpreted",
    )
    parser.add_argument(
        "--lower",
        nargs=3,
        default=("20", "80", "40"),
        metavar=("C0", "C1", "C2"),
        help="inclusive lower threshold",
    )
    parser.add_argument(
        "--upper",
        nargs=3,
        default=("45", "255", "255"),
        metavar=("C0", "C1", "C2"),
        help="inclusive upper threshold",
    )
    parser.add_argument("--output-dir", type=Path, help="directory for mask and result PNGs")
    parser.add_argument("--no-display", action="store_true", help="do not open GUI windows")
    parser.add_argument("--validate", action="store_true", help="run deterministic checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run segmentation and save reproducible outputs."""

    arguments = _build_parser().parse_args(argv)
    if arguments.validate:
        run_validation(arguments.output_dir)
        return 0

    lower = _triplet(arguments.lower)
    upper = _triplet(arguments.upper)
    input_path = resolve_input_path(arguments.input)
    image = read_bgr(input_path)
    mask, result = segment_image(image, arguments.space, lower, upper)
    foreground_pixels = int(cv2.countNonZero(mask))
    print(f"Input: {input_path}")
    print(f"Color space: {arguments.space}")
    print(f"Foreground pixels: {foreground_pixels}/{mask.size}")

    if arguments.output_dir is not None:
        mask_path, result_path = output_paths(
            arguments.output_dir, input_path.stem, arguments.space
        )
        write_image(mask_path, mask)
        write_image(result_path, result)
        print(f"Wrote: {mask_path}")
        print(f"Wrote: {result_path}")

    if not arguments.no_display:
        cv2.imshow("Input", image)
        cv2.imshow("Mask", mask)
        cv2.imshow(f"{arguments.space} segmentation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fill enclosed holes in a binary foreground mask with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def threshold_foreground(gray: Image, threshold_value: int = 220) -> Image:
    """Convert a grayscale image to a white-foreground binary mask."""
    if (
        gray is None
        or gray.size == 0
        or gray.ndim != 2
        or gray.dtype != np.uint8
    ):
        raise ValueError(
            "threshold_foreground expects a non-empty uint8 grayscale image"
        )
    if not 0 <= threshold_value <= 255:
        raise ValueError("threshold_value must be in the range [0, 255]")

    _, binary = cv2.threshold(
        gray, threshold_value, 255, cv2.THRESH_BINARY_INV
    )
    return binary


def fill_holes(binary: Image) -> tuple[Image, Image, Image]:
    """Return the filled mask, flooded background, and isolated hole mask.

    A one-pixel black border guarantees that the flood-fill seed belongs to
    the exterior background, even when foreground pixels touch the source
    image boundary.
    """
    if (
        binary is None
        or binary.size == 0
        or binary.ndim != 2
        or binary.dtype != np.uint8
    ):
        raise ValueError("fill_holes expects a non-empty uint8 single-channel mask")

    values = np.unique(binary)
    if not np.all(np.isin(values, (0, 255))):
        raise ValueError("fill_holes expects a binary mask containing only 0 and 255")

    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded_padded = padded.copy()
    height, width = flooded_padded.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flooded_padded, flood_mask, (0, 0), 255)

    flooded_background = flooded_padded[1:-1, 1:-1]
    holes = cv2.bitwise_not(flooded_background)
    filled = cv2.bitwise_or(binary, holes)
    return filled, flooded_background, holes


def run_pipeline(
    input_path: Path, threshold_value: int = 220
) -> tuple[Image, Image, Image, Image]:
    """Read an image, threshold it, and fill enclosed background regions."""
    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    binary = threshold_foreground(gray, threshold_value)
    filled, flooded_background, holes = fill_holes(binary)
    return binary, flooded_background, holes, filled


def write_outputs(
    output_directory: Path,
    binary: Image,
    flooded_background: Image,
    holes: Image,
    filled: Image,
) -> None:
    """Write every stage so the headless CLI is easy to inspect and test."""
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "01-thresholded.png": binary,
        "02-flooded-background.png": flooded_background,
        "03-holes.png": holes,
        "04-filled.png": filled,
    }
    for filename, image in outputs.items():
        path = output_directory / filename
        if not cv2.imwrite(str(path), image):
            raise OSError(f"OpenCV could not write output image: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill holes in a thresholded foreground mask with OpenCV."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("nickel.jpg"),
        help="Input image (default: bundled nickel.jpg)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=220,
        help="Binary-inverse threshold in [0, 255] (default: 220)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for the four pipeline images (default: output)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open GUI windows after writing the output images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stages = run_pipeline(args.image, args.threshold)
        write_outputs(args.output_dir, *stages)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 2

    binary, _, holes, filled = stages
    added_pixels = int(cv2.countNonZero(holes))
    print(f"input={args.image}")
    print(f"threshold={args.threshold}")
    print(f"foreground_pixels_before={int(cv2.countNonZero(binary))}")
    print(f"filled_hole_pixels={added_pixels}")
    print(f"foreground_pixels_after={int(cv2.countNonZero(filled))}")
    print(f"outputs={args.output_dir.resolve()}")

    if args.display:
        for label, image in zip(
            ("Thresholded", "Flooded exterior", "Holes", "Filled"), stages
        ):
            cv2.imshow(label, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Calculate the seven Hu moment invariants for binary shape images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def read_binary_image(image_path: str | Path, threshold: int = 128) -> np.ndarray:
    """Load an image as grayscale and threshold it to values 0 and 255."""
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def calculate_hu_moments(binary_image: np.ndarray) -> np.ndarray:
    """Return the raw Hu moments as a flat, seven-element array."""
    moments = cv2.moments(binary_image)
    return cv2.HuMoments(moments).reshape(7)


def log_transform_hu_moments(hu_moments: Sequence[float]) -> np.ndarray:
    """Apply the signed base-10 log transform, handling exact zeros safely."""
    transformed = []
    for value in hu_moments:
        scalar = float(value)
        transformed.append(
            0.0
            if scalar == 0.0
            else -math.copysign(1.0, scalar) * math.log10(abs(scalar))
        )
    return np.asarray(transformed, dtype=np.float64)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate Hu moments for one or more shape images."
    )
    parser.add_argument("images", nargs="+", type=Path, help="input image path(s)")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print raw Hu moments instead of signed log-transformed values",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        choices=range(256),
        metavar="0-255",
        help="binary threshold (default: 128)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for image_path in args.images:
        try:
            binary = read_binary_image(image_path, args.threshold)
        except FileNotFoundError as error:
            parser.error(str(error))

        values = calculate_hu_moments(binary)
        if not args.raw:
            values = log_transform_hu_moments(values)

        formatted = " ".join(f"{value:.5f}" for value in values)
        print(f"{image_path}: {formatted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

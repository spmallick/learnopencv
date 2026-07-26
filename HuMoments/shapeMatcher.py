#!/usr/bin/env python3
"""Compare binary shapes with OpenCV's Hu-moment-based matchShapes API."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from HuMoments import read_binary_image


EXAMPLE_DIR = Path(__file__).resolve().parent / "images"


def shape_distance(
    first: np.ndarray,
    second: np.ndarray,
    method: int = cv2.CONTOURS_MATCH_I2,
) -> float:
    """Return the OpenCV shape distance between two binary images."""
    return float(cv2.matchShapes(first, second, method, 0.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a reference shape with two other shapes."
    )
    parser.add_argument(
        "reference",
        nargs="?",
        type=Path,
        default=EXAMPLE_DIR / "S0.png",
        help="reference shape (default: images/S0.png)",
    )
    parser.add_argument(
        "different",
        nargs="?",
        type=Path,
        default=EXAMPLE_DIR / "K0.png",
        help="different shape (default: images/K0.png)",
    )
    parser.add_argument(
        "transformed",
        nargs="?",
        type=Path,
        default=EXAMPLE_DIR / "S4.png",
        help="transformed reference shape (default: images/S4.png)",
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

    try:
        reference = read_binary_image(args.reference, args.threshold)
        different = read_binary_image(args.different, args.threshold)
        transformed = read_binary_image(args.transformed, args.threshold)
    except FileNotFoundError as error:
        parser.error(str(error))

    comparisons = (
        (args.reference, args.reference, shape_distance(reference, reference)),
        (args.reference, args.different, shape_distance(reference, different)),
        (args.reference, args.transformed, shape_distance(reference, transformed)),
    )

    print("Shape distances")
    print("---------------")
    for first_path, second_path, distance in comparisons:
        print(f"{first_path.name} and {second_path.name}: {distance:.12f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

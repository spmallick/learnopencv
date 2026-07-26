#!/usr/bin/env python3
"""Generate a DICT_6X6_250 ArUco marker with current OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2 as cv
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def generate_marker(
    marker_id: int = 33,
    size: int = 200,
    border_bits: int = 1,
) -> np.ndarray:
    if not 0 <= marker_id < 250:
        raise ValueError("marker_id must be between 0 and 249.")
    if size < 32:
        raise ValueError("size must be at least 32 pixels.")
    if border_bits < 1:
        raise ValueError("border_bits must be at least 1.")

    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    return dictionary.generateImageMarker(
        marker_id, size, borderBits=border_bits
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 6x6 ArUco marker.")
    parser.add_argument("--id", type=int, default=33, help="Marker ID from 0 to 249.")
    parser.add_argument(
        "--size", type=int, default=200, help="Square image size in pixels."
    )
    parser.add_argument(
        "--border-bits",
        type=int,
        default=1,
        help="Marker border width in bits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "marker33.png",
        help="Output PNG path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        marker = generate_marker(args.id, args.size, args.border_bits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(args.output), marker):
            raise OSError(f"OpenCV could not write: {args.output}")
    except (OSError, ValueError, cv.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print(f"Wrote marker {args.id} ({args.size}x{args.size}) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

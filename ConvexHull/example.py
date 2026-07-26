#!/usr/bin/env python3
"""Find contours and draw their convex hulls with current OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def build_convex_hull_visualization(
    image: np.ndarray,
    threshold_value: int = 200,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return a drawing plus the contours and convex hulls used to create it."""
    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")
    if not 0 <= threshold_value <= 255:
        raise ValueError("threshold_value must be between 0 and 255.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (3, 3))
    _, binary = cv2.threshold(
        blurred, threshold_value, 255, cv2.THRESH_BINARY
    )

    # OpenCV 4 and 5 return (contours, hierarchy), not the OpenCV 3
    # three-value signature used by the original example.
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hulls = [cv2.convexHull(contour, clockwise=False) for contour in contours]

    drawing = np.zeros((*binary.shape, 3), dtype=np.uint8)
    for index in range(len(contours)):
        cv2.drawContours(
            drawing,
            contours,
            index,
            (0, 255, 0),
            2,
            cv2.LINE_8,
            hierarchy,
        )
        cv2.drawContours(
            drawing,
            hulls,
            index,
            (255, 255, 255),
            2,
            cv2.LINE_8,
        )
    return drawing, contours, hulls


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw image contours in green and their convex hulls in white."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "sample.jpg",
        help="Input image (default: bundled sample.jpg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "convex-hull-output.jpg",
        help="Output visualization path.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Binary threshold from 0 to 255 (default: 200).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the source and result in GUI windows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")
        drawing, contours, _ = build_convex_hull_visualization(
            source, args.threshold
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), drawing):
            raise OSError(f"OpenCV could not write: {args.output}")
    except (FileNotFoundError, OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"Detected {len(contours)} contours and wrote {args.output}")
    if args.display:
        cv2.imshow("Source", source)
        cv2.imshow("Contours and convex hulls", drawing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Correct perspective by mapping four image points to a rectangle."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    get_four_points,
    parse_points,
    read_image,
    rectify_image,
    show_images,
    write_image,
)


ASSET_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ASSET_DIR / "book1.jpg"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("perspective-corrected.jpg"),
    )
    parser.add_argument(
        "--points",
        type=parse_points,
        help="Source quad as x,y;x,y;x,y;x,y in clockwise order.",
    )
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = read_image(args.input)
    source_points = args.points
    if source_points is None:
        print(
            "Click the book corners in clockwise order, starting at the "
            "top-left. Press Enter after the fourth point."
        )
        source_points = get_four_points(source)

    corrected, _ = rectify_image(
        source, source_points, args.width, args.height
    )
    write_image(args.output, corrected)
    print(f"Saved {args.width}x{args.height} rectified image to {args.output.resolve()}")

    if args.display:
        show_images({"Source": source, "Perspective corrected": corrected})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


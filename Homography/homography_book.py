#!/usr/bin/env python3
"""Estimate a homography from known book corners and warp the source."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils import (
    compute_homography,
    parse_points,
    read_image,
    show_images,
    validate_quad,
    write_image,
)


ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_POINTS = "141,131;480,159;493,630;64,601"
DEFAULT_DESTINATION_POINTS = "318,256;534,372;316,670;73,473"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ASSET_DIR / "book2.jpg"
    )
    parser.add_argument(
        "--destination", type=Path, default=ASSET_DIR / "book1.jpg"
    )
    parser.add_argument(
        "--source-points",
        type=parse_points,
        default=parse_points(DEFAULT_SOURCE_POINTS),
    )
    parser.add_argument(
        "--destination-points",
        type=parse_points,
        default=parse_points(DEFAULT_DESTINATION_POINTS),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("warped-book.jpg")
    )
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = read_image(args.source)
    destination = read_image(args.destination)
    source_points = validate_quad(
        args.source_points, source.shape, name="source_points"
    )
    destination_points = validate_quad(
        args.destination_points,
        destination.shape,
        name="destination_points",
    )

    homography = compute_homography(source_points, destination_points)
    warped = cv2.warpPerspective(
        source,
        homography,
        (destination.shape[1], destination.shape[0]),
    )
    write_image(args.output, warped)
    print(f"Saved warped source image to {args.output.resolve()}")

    if args.display:
        show_images(
            {
                "Source": source,
                "Destination": destination,
                "Warped source": warped,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

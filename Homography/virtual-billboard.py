#!/usr/bin/env python3
"""Place a source image onto a four-corner region in another image."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    composite_on_quad,
    get_four_points,
    parse_points,
    read_image,
    show_images,
    write_image,
)


ASSET_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ASSET_DIR / "first-image.jpg"
    )
    parser.add_argument(
        "--destination", type=Path, default=ASSET_DIR / "times-square.jpg"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("virtual-billboard.jpg")
    )
    parser.add_argument(
        "--points",
        type=parse_points,
        help="Destination quad as x,y;x,y;x,y;x,y in clockwise order.",
    )
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = read_image(args.source)
    destination = read_image(args.destination)
    destination_points = args.points
    if destination_points is None:
        print(
            "Click the billboard corners in clockwise order, starting at "
            "the top-left. Press Enter after the fourth point."
        )
        destination_points = get_four_points(destination)

    result, _ = composite_on_quad(
        source, destination, destination_points
    )
    write_image(args.output, result)
    print(f"Saved billboard composite to {args.output.resolve()}")

    if args.display:
        show_images(
            {
                "Source": source,
                "Destination": destination,
                "Composite": result,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

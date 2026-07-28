#!/usr/bin/env python3
"""Crop a validated ROI and split an image into deterministic patches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "assets" / "sample-scene.png"


@dataclass(frozen=True)
class Tile:
    row: int
    column: int
    x: int
    y: int
    image: np.ndarray


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    return image


def validate_roi(
    image: np.ndarray, x: int, y: int, width: int, height: int
) -> tuple[slice, slice]:
    if image.ndim < 2 or image.size == 0:
        raise ValueError("image must be non-empty")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("x/y must be non-negative and width/height must be positive")
    image_height, image_width = image.shape[:2]
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI ({x}, {y}, {width}, {height}) exceeds "
            f"image bounds {image_width}x{image_height}"
        )
    return slice(y, y + height), slice(x, x + width)


def crop_image(
    image: np.ndarray, x: int, y: int, width: int, height: int
) -> np.ndarray:
    rows, columns = validate_roi(image, x, y, width, height)
    return image[rows, columns].copy()


def extract_tiles(
    image: np.ndarray,
    tile_width: int,
    tile_height: int,
    include_partial: bool = True,
) -> list[Tile]:
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    image_height, image_width = image.shape[:2]
    tiles: list[Tile] = []
    for row, y in enumerate(range(0, image_height, tile_height)):
        for column, x in enumerate(range(0, image_width, tile_width)):
            width = min(tile_width, image_width - x)
            height = min(tile_height, image_height - y)
            if not include_partial and (width != tile_width or height != tile_height):
                continue
            tiles.append(
                Tile(
                    row=row,
                    column=column,
                    x=x,
                    y=y,
                    image=image[y : y + height, x : x + width].copy(),
                )
            )
    return tiles


def make_crop_comparison(
    image: np.ndarray, cropped: np.ndarray, x: int, y: int, width: int, height: int
) -> np.ndarray:
    source = image.copy()
    cv2.rectangle(source, (x, y), (x + width - 1, y + height - 1), (0, 255, 255), 4)
    crop_panel = np.zeros_like(image)
    scale = min(image.shape[1] / cropped.shape[1], image.shape[0] / cropped.shape[0])
    scaled_size = (
        max(1, round(cropped.shape[1] * scale)),
        max(1, round(cropped.shape[0] * scale)),
    )
    resized = cv2.resize(cropped, scaled_size, interpolation=cv2.INTER_NEAREST)
    x0 = (crop_panel.shape[1] - resized.shape[1]) // 2
    y0 = (crop_panel.shape[0] - resized.shape[0]) // 2
    crop_panel[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    panels = [source, crop_panel]
    for panel, label in zip(panels, ["Validated ROI", "Cropped pixels"]):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return cv2.hconcat(panels)


def make_tile_contact_sheet(
    tiles: list[Tile], tile_width: int, tile_height: int
) -> np.ndarray:
    if not tiles:
        raise ValueError("at least one tile is required")
    rows = max(tile.row for tile in tiles) + 1
    columns = max(tile.column for tile in tiles) + 1
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for tile in tiles:
        y0 = tile.row * tile_height
        x0 = tile.column * tile_width
        h, w = tile.image.shape[:2]
        sheet[y0 : y0 + h, x0 : x0 + w] = tile.image
        cv2.rectangle(
            sheet,
            (x0, y0),
            (x0 + tile_width - 1, y0 + tile_height - 1),
            (255, 255, 255),
            1,
        )
    return sheet


def write_outputs(
    image: np.ndarray,
    output_dir: Path,
    x: int,
    y: int,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
) -> tuple[np.ndarray, list[Tile]]:
    cropped = crop_image(image, x, y, width, height)
    tiles = extract_tiles(image, tile_width, tile_height)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_dir / "cropped.png": cropped,
        output_dir / "crop-comparison.png": make_crop_comparison(
            image, cropped, x, y, width, height
        ),
        output_dir / "tile-contact-sheet.png": make_tile_contact_sheet(
            tiles, tile_width, tile_height
        ),
    }
    for tile in tiles:
        outputs[patch_dir / f"patch-r{tile.row}-c{tile.column}.png"] = tile.image
    for path, output in outputs.items():
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"Could not write output image: {path}")
    return cropped, tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        default=(160, 90, 320, 240))
    parser.add_argument("--tile-size", nargs=2, type=int, metavar=("W", "H"),
                        default=(160, 140))
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_bgr(args.input)
    x, y, width, height = args.roi
    tile_width, tile_height = args.tile_size
    cropped, tiles = write_outputs(
        image, args.output_dir, x, y, width, height, tile_width, tile_height
    )
    print(
        f"input={image.shape[1]}x{image.shape[0]} "
        f"crop={cropped.shape[1]}x{cropped.shape[0]} tiles={len(tiles)}"
    )
    if args.display:
        cv2.imshow(
            "Crop and ROI",
            make_crop_comparison(image, cropped, x, y, width, height),
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

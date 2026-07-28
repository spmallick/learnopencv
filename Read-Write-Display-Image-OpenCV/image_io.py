#!/usr/bin/env python3
"""Read, optionally display, and write images safely with OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "assets" / "sample-scene.png"
READ_MODES = {
    "color": cv2.IMREAD_COLOR,
    "grayscale": cv2.IMREAD_GRAYSCALE,
    "unchanged": cv2.IMREAD_UNCHANGED,
}


def read_image(path: Path, mode: str = "color") -> np.ndarray:
    if mode not in READ_MODES:
        raise ValueError(f"mode must be one of {', '.join(READ_MODES)}")
    image = cv2.imread(str(path), READ_MODES[mode])
    if image is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    if image.size == 0:
        raise ValueError("refusing to write an empty image")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, parameters or []):
        raise RuntimeError(f"Could not encode image: {path}")


def make_comparison(color: np.ndarray, gray: np.ndarray) -> np.ndarray:
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panels = [color.copy(), gray_bgr]
    for panel, label in zip(panels, ["IMREAD_COLOR", "IMREAD_GRAYSCALE"]):
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


def run_image_io(input_path: Path, output_dir: Path) -> dict[str, float | int]:
    color = read_image(input_path, "color")
    gray = read_image(input_path, "grayscale")
    unchanged = read_image(input_path, "unchanged")

    png_path = output_dir / "lossless-copy.png"
    gray_path = output_dir / "grayscale.png"
    jpeg_path = output_dir / "quality-90.jpg"
    comparison_path = output_dir / "image-io-comparison.png"
    write_image(png_path, color, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    write_image(gray_path, gray)
    write_image(jpeg_path, color, [cv2.IMWRITE_JPEG_QUALITY, 90])
    write_image(comparison_path, make_comparison(color, gray))

    png_roundtrip = read_image(png_path, "color")
    jpeg_roundtrip = read_image(jpeg_path, "color")
    if not np.array_equal(color, png_roundtrip):
        raise AssertionError("PNG round trip unexpectedly changed pixel values")
    jpeg_mae = float(
        np.mean(np.abs(color.astype(np.int16) - jpeg_roundtrip.astype(np.int16)))
    )
    return {
        "width": color.shape[1],
        "height": color.shape[0],
        "color_channels": color.shape[2],
        "unchanged_channels": 1 if unchanged.ndim == 2 else unchanged.shape[2],
        "jpeg_mae": jpeg_mae,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_image_io(args.input, args.output_dir)
    print(
        f"size={metrics['width']}x{metrics['height']} "
        f"channels={metrics['color_channels']} "
        f"jpeg_mae={metrics['jpeg_mae']:.3f}"
    )
    if args.display:
        color = read_image(args.input, "color")
        gray = read_image(args.input, "grayscale")
        cv2.imshow("Read and display images", make_comparison(color, gray))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

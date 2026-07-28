#!/usr/bin/env python3
"""Sobel and Canny edge detection with reproducible, headless outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "assets" / "sample-scene.png"


@dataclass(frozen=True)
class EdgeResult:
    gray: np.ndarray
    blurred: np.ndarray
    sobel_x: np.ndarray
    sobel_y: np.ndarray
    magnitude: np.ndarray
    canny: np.ndarray


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    return image


def normalize_gradient(gradient: np.ndarray) -> np.ndarray:
    return cv2.normalize(
        np.abs(gradient), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )


def detect_edges(
    image_bgr: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_size: int = 5,
) -> EdgeResult:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a three-channel BGR image")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must use uint8 pixels")
    if not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 255")
    if blur_size < 3 or blur_size % 2 == 0:
        raise ValueError("blur_size must be an odd integer of at least 3")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    sobel_x_f = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y_f = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude_f = cv2.magnitude(sobel_x_f, sobel_y_f)

    return EdgeResult(
        gray=gray,
        blurred=blurred,
        sobel_x=normalize_gradient(sobel_x_f),
        sobel_y=normalize_gradient(sobel_y_f),
        magnitude=normalize_gradient(magnitude_f),
        canny=cv2.Canny(blurred, low_threshold, high_threshold, L2gradient=True),
    )


def make_comparison(image_bgr: np.ndarray, result: EdgeResult) -> np.ndarray:
    panels = [
        image_bgr.copy(),
        cv2.cvtColor(result.magnitude, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(result.canny, cv2.COLOR_GRAY2BGR),
    ]
    labels = ["Input", "Sobel magnitude", "Canny"]
    for panel, label in zip(panels, labels):
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


def write_results(
    output_dir: Path, image_bgr: np.ndarray, result: EdgeResult
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gray": output_dir / "gray.png",
        "sobel_x": output_dir / "sobel-x.png",
        "sobel_y": output_dir / "sobel-y.png",
        "sobel_magnitude": output_dir / "sobel-magnitude.png",
        "canny": output_dir / "canny.png",
        "comparison": output_dir / "edge-comparison.png",
    }
    images = {
        "gray": result.gray,
        "sobel_x": result.sobel_x,
        "sobel_y": result.sobel_y,
        "sobel_magnitude": result.magnitude,
        "canny": result.canny,
        "comparison": make_comparison(image_bgr.copy(), result),
    }
    for name, path in outputs.items():
        if not cv2.imwrite(str(path), images[name]):
            raise RuntimeError(f"Could not write output image: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--low", type=int, default=100)
    parser.add_argument("--high", type=int, default=200)
    parser.add_argument("--blur-size", type=int, default=5)
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_bgr(args.input)
    result = detect_edges(image, args.low, args.high, args.blur_size)
    outputs = write_results(args.output_dir, image, result)
    print(
        f"size={image.shape[1]}x{image.shape[0]} "
        f"canny_pixels={cv2.countNonZero(result.canny)} "
        f"outputs={len(outputs)}"
    )
    if args.display:
        cv2.imshow("Sobel and Canny edge detection", make_comparison(image, result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

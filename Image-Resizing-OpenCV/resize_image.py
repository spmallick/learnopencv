#!/usr/bin/env python3
"""Resize images with explicit geometry and interpolation choices."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "assets" / "sample-scene.png"


@dataclass(frozen=True)
class LetterboxResult:
    image: np.ndarray
    scale: float
    offset_x: int
    offset_y: int
    content_width: int
    content_height: int


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    return image


def resize_exact(
    image: np.ndarray, width: int, height: int, interpolation: int
) -> np.ndarray:
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_by_scale(
    image: np.ndarray, scale_x: float, scale_y: float, interpolation: int
) -> np.ndarray:
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("scale factors must be positive")
    return cv2.resize(
        image, None, fx=scale_x, fy=scale_y, interpolation=interpolation
    )


def rounded_positive_size(value: float) -> int:
    """Round a positive size half up, matching C++ std::lround."""
    return max(1, int(value + 0.5))


def resize_to_fit(
    image: np.ndarray,
    max_width: int,
    max_height: int,
    allow_upscale: bool = False,
) -> tuple[np.ndarray, float]:
    if image.size == 0:
        raise ValueError("image must be non-empty")
    if max_width <= 0 or max_height <= 0:
        raise ValueError("bounding-box dimensions must be positive")
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    target = (
        rounded_positive_size(width * scale),
        rounded_positive_size(height * scale),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, target, interpolation=interpolation), scale


def letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    color: tuple[int, int, int] = (32, 32, 32),
) -> LetterboxResult:
    resized, scale = resize_to_fit(
        image, target_width, target_height, allow_upscale=True
    )
    output = np.full((target_height, target_width, 3), color, dtype=image.dtype)
    offset_x = (target_width - resized.shape[1]) // 2
    offset_y = (target_height - resized.shape[0]) // 2
    output[
        offset_y : offset_y + resized.shape[0],
        offset_x : offset_x + resized.shape[1],
    ] = resized
    return LetterboxResult(
        image=output,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        content_width=resized.shape[1],
        content_height=resized.shape[0],
    )


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
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
    return panel


def make_comparison(image: np.ndarray) -> np.ndarray:
    size = (image.shape[1] // 2, image.shape[0] // 2)
    area_small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    linear_small = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    area_restored = cv2.resize(
        area_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    linear_restored = cv2.resize(
        linear_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return cv2.hconcat(
        [
            labeled_panel(image, "Input"),
            labeled_panel(area_restored, "Downscale: INTER_AREA"),
            labeled_panel(linear_restored, "Downscale: INTER_LINEAR"),
        ]
    )


def run_resize_demo(input_path: Path, output_dir: Path) -> dict[str, float | int]:
    image = read_bgr(input_path)
    down_area = resize_exact(
        image, image.shape[1] // 2, image.shape[0] // 2, cv2.INTER_AREA
    )
    up_linear = resize_by_scale(image, 1.5, 1.5, cv2.INTER_LINEAR)
    up_cubic = resize_by_scale(image, 1.5, 1.5, cv2.INTER_CUBIC)
    boxed = letterbox(image, 640, 640)
    outputs = {
        output_dir / "downscale-inter-area.png": down_area,
        output_dir / "upscale-inter-linear.png": up_linear,
        output_dir / "upscale-inter-cubic.png": up_cubic,
        output_dir / "letterbox-640.png": boxed.image,
        output_dir / "resize-comparison.png": make_comparison(image),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, output in outputs.items():
        if not cv2.imwrite(str(path), output):
            raise RuntimeError(f"Could not write output image: {path}")
    return {
        "input_width": image.shape[1],
        "input_height": image.shape[0],
        "down_width": down_area.shape[1],
        "down_height": down_area.shape[0],
        "letterbox_scale": boxed.scale,
        "letterbox_content_width": boxed.content_width,
        "letterbox_content_height": boxed.content_height,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_resize_demo(args.input, args.output_dir)
    print(
        f"input={metrics['input_width']}x{metrics['input_height']} "
        f"down={metrics['down_width']}x{metrics['down_height']} "
        f"letterbox_content={metrics['letterbox_content_width']}x"
        f"{metrics['letterbox_content_height']}"
    )
    if args.display:
        cv2.imshow("Resize interpolation comparison", make_comparison(read_bgr(args.input)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

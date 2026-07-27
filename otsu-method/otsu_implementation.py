"""A small, tested implementation of Otsu's global thresholding method."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "boat.jpg"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "otsu-custom.png"


def _validate_grayscale(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Otsu thresholding expects a non-empty uint8 grayscale image.")


def otsu_threshold(image: np.ndarray, *, normalize_histogram: bool = False) -> int:
    """Return the lowest 8-bit threshold that maximizes between-class variance.

    Empty classes are excluded from the search. A constant image has no valid
    two-class split, so this function returns 0, matching OpenCV's behavior.
    """

    _validate_grayscale(image)

    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    if normalize_histogram:
        histogram /= histogram.sum()

    intensities = np.arange(256, dtype=np.float64)
    background_weight = np.cumsum(histogram)
    foreground_weight = histogram.sum() - background_weight
    background_sum = np.cumsum(histogram * intensities)
    foreground_sum = background_sum[-1] - background_sum

    valid = (background_weight > 0) & (foreground_weight > 0)
    if not np.any(valid):
        return 0

    background_mean = np.zeros(256, dtype=np.float64)
    foreground_mean = np.zeros(256, dtype=np.float64)
    np.divide(
        background_sum,
        background_weight,
        out=background_mean,
        where=background_weight > 0,
    )
    np.divide(
        foreground_sum,
        foreground_weight,
        out=foreground_mean,
        where=foreground_weight > 0,
    )

    variance = np.full(256, -np.inf, dtype=np.float64)
    mean_delta = background_mean[valid] - foreground_mean[valid]
    variance[valid] = (
        background_weight[valid] * foreground_weight[valid] * mean_delta * mean_delta
    )
    return int(np.argmax(variance))


def apply_otsu(
    image: np.ndarray,
    *,
    reduce_noise: bool = False,
    normalize_histogram: bool = False,
) -> tuple[int, np.ndarray]:
    """Threshold an image with the custom implementation."""

    _validate_grayscale(image)
    processed = cv2.GaussianBlur(image, (5, 5), 0) if reduce_noise else image
    threshold = otsu_threshold(
        processed, normalize_histogram=normalize_histogram
    )
    binary = np.where(processed > threshold, 255, 0).astype(np.uint8)
    return threshold, binary


def otsu_implementation(
    img_title: str | Path = DEFAULT_INPUT,
    is_normalized: bool = False,
    is_reduce_noise: bool = False,
) -> int:
    """Backward-compatible file-based wrapper used by the original tutorial."""

    image_path = Path(img_title).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {image_path}")
    threshold, _ = apply_otsu(
        image,
        reduce_noise=is_reduce_noise,
        normalize_histogram=is_normalized,
    )
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a from-scratch implementation of Otsu thresholding."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--blur",
        action="store_true",
        help="Apply a 5x5 Gaussian blur before choosing the threshold.",
    )
    parser.add_argument(
        "--normalize-histogram",
        action="store_true",
        help="Use probabilities instead of counts; the threshold is unchanged.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {input_path}")

    threshold, binary = apply_otsu(
        image,
        reduce_noise=args.blur,
        normalize_histogram=args.normalize_histogram,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), binary):
        raise OSError(f"Could not write output image: {output_path}")

    print(f"Custom Otsu threshold: {threshold}")
    print(f"Saved binary image: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

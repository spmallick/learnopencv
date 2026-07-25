#!/usr/bin/env python3
"""Reconstruct a color image by ECC-aligning stacked monochrome channels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ecc_utils import MOTION_MODELS, align_image


SCRIPT_DIR = Path(__file__).resolve().parent


def split_stacked_channels(image: np.ndarray) -> list[np.ndarray]:
    """Split a vertically stacked B/G/R plate into equal-size channels."""
    height = image.shape[0] // 3
    if height < 1:
        raise ValueError("Input is too short to contain three stacked channels")
    cropped = image[: height * 3]
    return [cropped[index * height : (index + 1) * height] for index in range(3)]


def align_stacked_channels(
    image: np.ndarray,
    motion_model: int = cv2.MOTION_HOMOGRAPHY,
    iterations: int = 5000,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, list[float], list[np.ndarray]]:
    """Align the blue and green plates to red, returning before/after composites."""
    channels = split_stacked_channels(image)
    unaligned = cv2.merge(channels)
    aligned_channels: list[np.ndarray] = []
    correlations: list[float] = []
    warps: list[np.ndarray] = []
    for index in range(2):
        correlation, warp, aligned = align_image(
            channels[2],
            channels[index],
            motion_model,
            iterations,
            epsilon,
            use_gradient=True,
        )
        aligned_channels.append(aligned)
        correlations.append(correlation)
        warps.append(warp)
    aligned_channels.append(channels[2].copy())
    return unaligned, cv2.merge(aligned_channels), correlations, warps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "images" / "emir.jpg")
    parser.add_argument(
        "--motion", choices=sorted(MOTION_MODELS), default="homography"
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read stacked image: {args.input}")
    unaligned, aligned, correlations, warps = align_stacked_channels(
        image,
        MOTION_MODELS[args.motion],
        args.iterations,
        args.epsilon,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.output_dir / "stacked-color-unaligned.jpg"
    after_path = args.output_dir / "stacked-color-aligned.jpg"
    if not cv2.imwrite(str(before_path), unaligned) or not cv2.imwrite(
        str(after_path), aligned
    ):
        raise RuntimeError("Could not write color reconstruction outputs")

    print(f"OpenCV: {cv2.__version__}")
    print(f"Motion model: {args.motion}")
    for index, (correlation, warp) in enumerate(zip(correlations, warps)):
        channel = "blue" if index == 0 else "green"
        print(f"{channel} ECC correlation: {correlation:.8f}")
        print(f"{channel} warp:\n{warp}")
    print(f"Saved: {after_path}")

    if args.validate:
        if (
            not all(np.isfinite(correlations))
            or min(correlations) <= 0.0
            or not all(np.isfinite(warp).all() for warp in warps)
            or float(aligned.std()) < 10.0
        ):
            raise RuntimeError("Stacked-channel ECC validation failed")
        print("Validation: PASS")

    if not args.no_display:
        cv2.imshow("Unaligned color", unaligned)
        cv2.imshow("ECC-aligned color", aligned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Detect and visualize blobs with OpenCV's SimpleBlobDetector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_IMAGE = Path(__file__).resolve().with_name("blob.jpg")


def create_detector() -> cv2.SimpleBlobDetector:
    """Create the detector used in the LearnOpenCV tutorial."""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.minRepeatability = 2
    params.minDistBetweenBlobs = 10

    params.filterByColor = True
    params.blobColor = 0

    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 5000

    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.maxCircularity = 1.0

    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.maxConvexity = 1.0

    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1.0

    return cv2.SimpleBlobDetector_create(params)


def read_grayscale(path: Path) -> np.ndarray:
    """Read an image as grayscale and fail with a useful error."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    return image


def detect_blobs(
    image: np.ndarray, detector: cv2.SimpleBlobDetector | None = None
) -> Sequence[cv2.KeyPoint]:
    """Detect blobs in a non-empty grayscale image."""
    if image is None or image.size == 0:
        raise ValueError("The input image must not be empty.")
    if image.ndim != 2:
        raise ValueError("detect_blobs expects a single-channel grayscale image.")
    return (detector or create_detector()).detect(image)


def draw_blobs(
    image: np.ndarray, keypoints: Sequence[cv2.KeyPoint]
) -> np.ndarray:
    """Draw each keypoint as a red circle whose radius represents blob size."""
    return cv2.drawKeypoints(
        image,
        list(keypoints),
        None,
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image (default: {DEFAULT_IMAGE.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("blob-keypoints.png"),
        help="Output visualization (default: blob-keypoints.png)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Also open the result in an OpenCV window.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    image = read_grayscale(args.input)
    keypoints = detect_blobs(image)
    visualization = draw_blobs(image, keypoints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), visualization):
        raise OSError(f"Could not write output image: {args.output}")

    print(f"Detected {len(keypoints)} blobs.")
    print(f"Saved visualization to {args.output.resolve()}")

    if args.display:
        cv2.imshow("Detected blobs", visualization)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

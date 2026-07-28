#!/usr/bin/env python3
"""Generate the deterministic sample image used by the tutorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_scene(width: int = 640, height: int = 420) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip(30 + 70 * x + 15 * y, 0, 255)
    image[..., 1] = np.clip(35 + 30 * x + 55 * y, 0, 255)
    image[..., 2] = np.clip(45 + 25 * x + 35 * y, 0, 255)

    cv2.rectangle(image, (55, 65), (275, 305), (210, 90, 35), -1)
    cv2.rectangle(image, (82, 92), (248, 278), (245, 180, 75), 5)
    cv2.circle(image, (430, 170), 92, (35, 145, 235), -1)
    cv2.circle(image, (430, 170), 55, (245, 225, 190), 6)
    points = np.array([[345, 335], [505, 285], [575, 365]], dtype=np.int32)
    cv2.fillConvexPoly(image, points, (95, 210, 115))
    cv2.line(image, (25, 365), (610, 45), (245, 245, 245), 4, cv2.LINE_AA)
    cv2.putText(
        image,
        "OpenCV",
        (290, 395),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "assets" / "sample-scene.png",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), build_scene()):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke test the bundled average-face dataset without opening a window."""

from pathlib import Path

import cv2
import numpy as np

import faceAverage


def main() -> int:
    images, points = faceAverage.read_dataset(Path(__file__).with_name("presidents"))
    output, triangle_count = faceAverage.create_average_face(
        images, points, width=320, height=320
    )
    assert output.shape == (320, 320, 3)
    assert np.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0
    assert float(output.std()) > 0.04
    assert triangle_count >= 50
    print(
        f"OpenCV {cv2.__version__}: PASS "
        f"({len(images)} faces, {triangle_count} triangles)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

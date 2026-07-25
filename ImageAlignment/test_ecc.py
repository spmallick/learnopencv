#!/usr/bin/env python3
"""Smoke tests for the pair and stacked-channel ECC examples."""

from pathlib import Path

import cv2
import numpy as np

from ecc_utils import align_image, mean_absolute_error
from image_alignment import align_stacked_channels


def main() -> int:
    image_dir = Path(__file__).with_name("images")
    template = cv2.imread(str(image_dir / "image1.jpg"))
    moving = cv2.imread(str(image_dir / "image2.jpg"))
    assert template is not None and moving is not None
    correlation, warp, aligned = align_image(
        template, moving, cv2.MOTION_EUCLIDEAN, 5000, 1e-7
    )
    assert np.isfinite(correlation) and correlation > 0.0
    assert np.isfinite(warp).all()
    moving_resized = cv2.resize(moving, (template.shape[1], template.shape[0]))
    assert mean_absolute_error(template, aligned) < mean_absolute_error(
        template, moving_resized
    )

    stacked = cv2.imread(str(image_dir / "emir.jpg"), cv2.IMREAD_GRAYSCALE)
    assert stacked is not None
    unaligned, color, correlations, warps = align_stacked_channels(stacked)
    assert color.shape == unaligned.shape
    assert min(correlations) > 0.0
    assert all(np.isfinite(warp).all() for warp in warps)
    print(
        f"OpenCV {cv2.__version__}: PASS "
        f"(pair CC {correlation:.6f}; stacked CC {min(correlations):.6f}+)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

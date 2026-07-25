#!/usr/bin/env python3
"""Small deterministic smoke test for the bundled head-pose example."""

from pathlib import Path

import cv2
import numpy as np

import headPose


def main() -> int:
    image = cv2.imread(str(Path(__file__).with_name("headPose.jpg")))
    assert image is not None
    camera, rotation, translation, rmse = headPose.estimate_pose(image)
    result = headPose.draw_pose(image, camera, rotation, translation)
    assert result.shape == image.shape
    assert np.isfinite(rotation).all()
    assert np.isfinite(translation).all()
    assert np.isfinite(rmse) and rmse < 50.0
    assert np.count_nonzero(cv2.absdiff(result, image)) > 0
    print(f"OpenCV {cv2.__version__}: PASS (RMSE {rmse:.3f} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

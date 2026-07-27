from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from Undistort import (  # noqa: E402
    calibrate_from_images,
    discover_images,
    undistort_image,
)


@pytest.fixture(scope="module")
def calibration():
    paths = discover_images(str(PROJECT_DIR / "images" / "*.jpg"))
    return calibrate_from_images(paths, require_all=True)


def test_all_41_checkerboards_calibrate_with_expected_geometry(calibration):
    assert len(calibration.successful_images) == 41
    assert calibration.failed_images == ()
    assert calibration.image_size == (640, 480)

    assert calibration.rms == pytest.approx(0.26032, abs=0.02)
    assert calibration.reprojection_rmse == pytest.approx(
        calibration.rms, abs=1e-5
    )
    assert calibration.camera_matrix[0, 0] == pytest.approx(503.51, abs=3.0)
    assert calibration.camera_matrix[1, 1] == pytest.approx(503.15, abs=3.0)
    assert calibration.camera_matrix[0, 2] == pytest.approx(313.41, abs=3.0)
    assert calibration.camera_matrix[1, 2] == pytest.approx(243.09, abs=3.0)
    assert calibration.distortion_coefficients.ravel()[0] == pytest.approx(
        0.211, abs=0.05
    )
    assert calibration.distortion_coefficients.ravel()[1] == pytest.approx(
        -0.484, abs=0.10
    )


def test_direct_and_remap_outputs_agree(calibration):
    image = cv2.imread(str(calibration.successful_images[0]))
    direct = undistort_image(image, calibration, method="direct")
    remapped = undistort_image(image, calibration, method="remap")

    assert direct.image.shape == image.shape
    assert direct.roi == pytest.approx((5, 7, 627, 466), abs=2)
    assert remapped.roi == direct.roi
    difference = cv2.absdiff(direct.image, remapped.image)
    assert int(difference.max()) <= 8
    assert float(difference.mean()) < 0.1


def test_crop_uses_valid_pixel_roi(calibration):
    image = cv2.imread(str(calibration.successful_images[0]))
    result = undistort_image(image, calibration, method="remap", crop=True)
    _, _, width, height = result.roi
    assert result.image.shape[:2] == (height, width)


@pytest.mark.parametrize("alpha", [-0.01, 1.01, np.nan])
def test_invalid_alpha_is_rejected(calibration, alpha):
    image = cv2.imread(str(calibration.successful_images[0]))
    with pytest.raises(ValueError, match="alpha"):
        undistort_image(image, calibration, alpha=alpha)


def test_invalid_inputs_are_rejected(calibration, tmp_path):
    image = cv2.imread(str(calibration.successful_images[0]))
    with pytest.raises(ValueError, match="method"):
        undistort_image(image, calibration, method="unknown")
    with pytest.raises(ValueError, match="empty"):
        undistort_image(np.empty((0, 0, 3), dtype=np.uint8), calibration)
    with pytest.raises(ValueError, match="does not match"):
        undistort_image(image[:100, :100], calibration)
    with pytest.raises(ValueError, match="At least one"):
        calibrate_from_images([])
    with pytest.raises(FileNotFoundError, match="No calibration images"):
        discover_images(str(tmp_path / "*.jpg"))


def test_mismatched_calibration_image_sizes_are_rejected(tmp_path):
    first = np.zeros((40, 50, 3), dtype=np.uint8)
    second = np.zeros((41, 50, 3), dtype=np.uint8)
    first_path = tmp_path / "first.jpg"
    second_path = tmp_path / "second.jpg"
    assert cv2.imwrite(str(first_path), first)
    assert cv2.imwrite(str(second_path), second)

    with pytest.raises(ValueError, match="expected"):
        calibrate_from_images([first_path, second_path])


def test_headless_cli_writes_reproducible_outputs(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "Undistort.py"),
            "--images",
            str(PROJECT_DIR / "images" / "*.jpg"),
            "--output-dir",
            str(tmp_path),
            "--require-all",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Checkerboards detected: 41/41" in completed.stdout
    assert "Image size: 640x480" in completed.stdout
    for filename in (
        "calibration-corners.jpg",
        "undistorted-direct.jpg",
        "undistorted-remap.jpg",
        "calibration.yml",
    ):
        path = tmp_path / filename
        assert path.is_file()
        assert path.stat().st_size > 0

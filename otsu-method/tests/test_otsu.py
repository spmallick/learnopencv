from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from otsu_implementation import apply_otsu, otsu_threshold  # noqa: E402


def assert_matches_opencv(image: np.ndarray) -> None:
    expected_threshold, expected_binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    threshold, binary = apply_otsu(image)
    assert threshold == int(expected_threshold)
    np.testing.assert_array_equal(binary, expected_binary)


def test_boat_matches_opencv() -> None:
    image = cv2.imread(str(PROJECT_DIR / "boat.jpg"), cv2.IMREAD_GRAYSCALE)
    assert image is not None
    threshold, binary = apply_otsu(image)
    assert threshold == 132
    expected_threshold, expected_binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    assert threshold == int(expected_threshold)
    np.testing.assert_array_equal(binary, expected_binary)


def test_synthetic_bimodal_image_matches_opencv() -> None:
    image = np.full((64, 64), 20, dtype=np.uint8)
    image[:, 32:] = 200
    assert_matches_opencv(image)
    assert otsu_threshold(image, normalize_histogram=True) == otsu_threshold(image)


@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_image_matches_opencv(value: int) -> None:
    assert_matches_opencv(np.full((16, 16), value, dtype=np.uint8))


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 0), dtype=np.uint8),
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8), dtype=np.float32),
    ],
)
def test_invalid_input_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        otsu_threshold(image)


def test_cli_saves_headless_output(tmp_path: Path) -> None:
    output = tmp_path / "binary.png"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "otsu_implementation.py"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Custom Otsu threshold: 132" in result.stdout
    assert output.is_file()

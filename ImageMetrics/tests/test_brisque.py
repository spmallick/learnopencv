from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_DIR / "Python"
sys.path.insert(0, str(PYTHON_DIR))

import brisquequality  # noqa: E402


def test_sample_score_regression() -> None:
    score = brisquequality.calculate_brisque_score(
        PROJECT_DIR / "Images" / "original-scaled-image.jpg"
    )

    assert score == pytest.approx(20.28, abs=0.03)


def test_blur_receives_worse_score(tmp_path: Path) -> None:
    source = cv2.imread(
        str(PROJECT_DIR / "Images" / "original-scaled-image.jpg"),
        cv2.IMREAD_COLOR,
    )
    assert source is not None

    blurred_path = tmp_path / "blurred.jpg"
    blurred = cv2.GaussianBlur(source, (31, 31), 0)
    assert cv2.imwrite(str(blurred_path), blurred)

    original_score = brisquequality.calculate_brisque_score(
        PROJECT_DIR / "Images" / "original-scaled-image.jpg"
    )
    blurred_score = brisquequality.calculate_brisque_score(blurred_path)

    assert blurred_score > original_score + 30.0


def test_missing_image_is_reported() -> None:
    with pytest.raises(FileNotFoundError, match="Could not read image"):
        brisquequality.calculate_brisque_score(PROJECT_DIR / "missing.jpg")


def test_cli_defaults_work_outside_project_directory(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PYTHON_DIR / "brisquequality.py"),
            str(PROJECT_DIR / "Images" / "original-scaled-image.jpg"),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "BRISQUE score: 20.2" in result.stdout

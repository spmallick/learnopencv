from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import HuMoments as hu  # noqa: E402
import shapeMatcher as matcher  # noqa: E402


def test_known_hu_moments_and_zero_safe_transform() -> None:
    image = hu.read_binary_image(PROJECT_DIR / "images" / "K0.png")
    raw = hu.calculate_hu_moments(image)
    transformed = hu.log_transform_hu_moments(raw)

    assert raw.shape == (7,)
    assert raw[0] == pytest.approx(0.0016266258, rel=1e-6)
    assert transformed[0] == pytest.approx(2.78871, abs=1e-5)
    assert np.isfinite(transformed).all()
    assert hu.log_transform_hu_moments([0.0])[0] == 0.0


def test_transformed_shape_is_closer_than_different_letter() -> None:
    reference = hu.read_binary_image(PROJECT_DIR / "images" / "S0.png")
    different = hu.read_binary_image(PROJECT_DIR / "images" / "K0.png")
    transformed = hu.read_binary_image(PROJECT_DIR / "images" / "S4.png")

    self_distance = matcher.shape_distance(reference, reference)
    different_distance = matcher.shape_distance(reference, different)
    transformed_distance = matcher.shape_distance(reference, transformed)

    assert self_distance == pytest.approx(0.0, abs=1e-12)
    assert different_distance == pytest.approx(0.1144026253, rel=1e-6)
    assert transformed_distance == pytest.approx(0.0134756130, rel=1e-6)
    assert transformed_distance < different_distance


def test_shape_matcher_defaults_work_outside_project_directory(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "shapeMatcher.py")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "S0.png and K0.png: 0.114402" in result.stdout

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest


PYTHON_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

from stereo_depth import (  # noqa: E402
    RectificationMaps,
    StereoBMConfig,
    compute_disparity,
    disparity_to_depth,
    find_largest_obstacle,
    fit_depth_model,
    load_config,
    load_rectification_maps,
    save_config,
)


def synthetic_pair(disparity: int = 8) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(7)
    left = random.integers(0, 256, (120, 192), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]
    right[:, -disparity:] = random.integers(
        0, 256, (left.shape[0], disparity), dtype=np.uint8
    )
    return left, right


def synthetic_config() -> StereoBMConfig:
    return StereoBMConfig(
        num_disparities=16,
        block_size=9,
        pre_filter_type=1,
        pre_filter_size=9,
        pre_filter_cap=31,
        texture_threshold=0,
        uniqueness_ratio=0,
        speckle_range=0,
        speckle_window_size=0,
        disp12_max_diff=-1,
        min_disparity=0,
        depth_scale=1000.0,
        depth_offset=5.0,
    )


def test_legacy_config_is_validated_and_migrated() -> None:
    path = PROJECT_DIR / "data" / "depth_estmation_params_py.xml"
    config = load_config(path)
    assert config.num_disparities == 144
    assert config.block_size == 35
    assert config.depth_scale == pytest.approx(36.51238116534359 * 144)
    assert config.depth_offset == 0.0


def test_config_round_trip_preserves_scale_and_offset(tmp_path: Path) -> None:
    expected = synthetic_config()
    path = save_config(expected, tmp_path / "config.xml")
    assert load_config(path) == expected


@pytest.mark.parametrize(
    "config",
    [
        replace(synthetic_config(), num_disparities=15),
        replace(synthetic_config(), block_size=8),
        replace(synthetic_config(), pre_filter_cap=0),
        replace(synthetic_config(), depth_scale=float("nan")),
    ],
)
def test_invalid_config_is_rejected(config: StereoBMConfig) -> None:
    with pytest.raises(ValueError):
        config.validated()


def test_rectification_maps_are_complete() -> None:
    maps = load_rectification_maps()
    assert maps.image_size == (640, 480)
    assert maps.left_x.shape[:2] == (480, 640)
    assert maps.right_x.shape[:2] == (480, 640)


def test_invalid_rectification_map_type_is_rejected() -> None:
    invalid = np.zeros((10, 10), dtype=np.float64)
    maps = RectificationMaps(invalid, invalid, invalid, invalid)
    with pytest.raises(ValueError, match="Rectification maps"):
        maps.validated()


def test_synthetic_stereobm_recovers_known_shift() -> None:
    left, right = synthetic_pair(disparity=8)
    disparity = compute_disparity(left, right, synthetic_config())
    assert np.mean(np.isfinite(disparity)) > 0.75
    assert float(np.nanmedian(disparity)) == pytest.approx(8.0, abs=0.25)


def test_depth_model_fit_preserves_slope_and_intercept() -> None:
    disparities = np.array([8.0, 10.0, 16.0, 25.0, 40.0])
    depths = 1200.0 / (disparities - 2.0) + 17.0
    scale, offset, rmse = fit_depth_model(
        disparities, depths, min_disparity=2.0
    )
    assert scale == pytest.approx(1200.0)
    assert offset == pytest.approx(17.0)
    assert rmse < 1e-10


def test_invalid_disparity_becomes_nan_depth() -> None:
    config = replace(synthetic_config(), min_disparity=2)
    disparity = np.array([[np.nan, -1.0, 2.0, 7.0]], dtype=np.float32)
    depth = disparity_to_depth(disparity, config)
    assert np.all(np.isnan(depth[0, :3]))
    assert depth[0, 3] == pytest.approx(205.0)


def test_obstacle_detection_is_deterministic() -> None:
    depth = np.full((100, 100), np.nan, dtype=np.float32)
    depth[20:70, 30:80] = 50.0
    depth[30:40, 40:50] = np.nan
    mask, obstacle = find_largest_obstacle(
        depth,
        min_depth=10.0,
        max_depth=100.0,
        minimum_area_fraction=0.1,
    )
    assert obstacle is not None
    assert obstacle.bounding_box == (30, 20, 50, 50)
    assert obstacle.mean_depth == pytest.approx(50.0)
    assert np.count_nonzero(mask) == 2400


def test_headless_cli_saves_outputs(tmp_path: Path) -> None:
    left, right = synthetic_pair()
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    assert cv2.imwrite(str(left_path), left)
    assert cv2.imwrite(str(right_path), right)
    config_path = save_config(synthetic_config(), tmp_path / "config.xml")
    output_dir = tmp_path / "results"

    result = subprocess.run(
        [
            sys.executable,
            str(PYTHON_DIR / "stereo_depth.py"),
            "--left",
            str(left_path),
            "--right",
            str(right_path),
            "--config",
            str(config_path),
            "--already-rectified",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "left-rectified.png",
        "right-rectified.png",
        "disparity.png",
        "depth.png",
    ):
        assert (output_dir / name).is_file()

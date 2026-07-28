from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from hog import (
    DEFAULT_CONFIG,
    cell_histograms,
    compute_descriptor,
    make_demo_image,
    prepare_image,
    run,
    visualize_hog,
)


def test_standard_descriptor_geometry() -> None:
    assert DEFAULT_CONFIG.descriptor_length == 3780
    descriptor = compute_descriptor(make_demo_image())
    assert descriptor.shape == (3780,)
    assert descriptor.dtype == np.float32
    assert np.isfinite(descriptor).all()
    assert np.linalg.norm(descriptor) > 1


def test_descriptor_is_deterministic() -> None:
    image = make_demo_image()
    first = compute_descriptor(image)
    second = compute_descriptor(image.copy())
    np.testing.assert_array_equal(first, second)


def test_horizontal_gradient_votes_into_zero_degree_bin() -> None:
    ramp = np.tile(np.arange(64, dtype=np.float32), (128, 1))
    histograms = cell_histograms(ramp)
    assert histograms[..., 0].sum() > 0
    assert np.count_nonzero(histograms[..., 1:]) == 0


def test_flat_image_and_visualization() -> None:
    flat = np.zeros((128, 64, 3), dtype=np.uint8)
    descriptor = compute_descriptor(flat)
    visualization = visualize_hog(flat)
    histograms = cell_histograms(flat)
    assert np.count_nonzero(descriptor) == 0
    assert np.count_nonzero(histograms) == 0
    assert visualization.shape == (512, 256, 3)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        prepare_image(np.array([], dtype=np.uint8))
    with pytest.raises(ValueError):
        prepare_image(np.zeros((5, 5, 2), dtype=np.uint8))


def test_headless_outputs(tmp_path: Path) -> None:
    summary = run(None, tmp_path)
    assert summary["descriptor_length"] == 3780
    for output in summary["outputs"].values():
        assert Path(output).is_file()
    assert cv2.imread(summary["outputs"]["visualization"]).shape == (512, 256, 3)

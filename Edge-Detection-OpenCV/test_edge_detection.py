from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from edge_detection import DEFAULT_INPUT, detect_edges, make_comparison, read_bgr


def test_synthetic_rectangle_has_thin_connected_edges() -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 24), (69, 67), (255, 255, 255), -1)
    original = image.copy()
    result = detect_edges(image, 50, 150, 5)
    comparison = make_comparison(image, result)
    edge_pixels = cv2.countNonZero(result.canny)
    assert result.canny.shape == image.shape[:2]
    assert result.canny.dtype == np.uint8
    assert 150 <= edge_pixels <= 250
    assert cv2.countNonZero(result.magnitude) > 0
    assert comparison.shape == (96, 96 * 3, 3)
    assert np.array_equal(image, original)


def test_threshold_and_input_validation() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="low < high"):
        detect_edges(image, 200, 100)
    with pytest.raises(ValueError, match="three-channel"):
        detect_edges(image[..., 0])


def test_cli_writes_all_documented_outputs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parent / "edge_detection.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(DEFAULT_INPUT),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "canny_pixels=" in completed.stdout
    expected = {
        "gray.png",
        "sobel-x.png",
        "sobel-y.png",
        "sobel-magnitude.png",
        "canny.png",
        "edge-comparison.png",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == expected
    assert read_bgr(DEFAULT_INPUT).shape == (420, 640, 3)


def test_missing_input_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not read input image"):
        read_bgr(tmp_path / "missing.png")

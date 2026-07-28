from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from resize_image import (
    DEFAULT_INPUT,
    letterbox,
    resize_by_scale,
    resize_exact,
    resize_to_fit,
)


def test_exact_and_scale_geometry() -> None:
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    assert resize_exact(image, 20, 10, cv2.INTER_AREA).shape == (10, 20, 3)
    assert resize_by_scale(image, 0.5, 0.5, cv2.INTER_AREA).shape == (20, 40, 3)


def test_aspect_fit_and_letterbox() -> None:
    image = np.full((40, 80, 3), (10, 20, 30), dtype=np.uint8)
    fitted, scale = resize_to_fit(image, 100, 100, allow_upscale=True)
    boxed = letterbox(image, 100, 100)
    assert scale == pytest.approx(1.25)
    assert fitted.shape == (50, 100, 3)
    assert boxed.image.shape == (100, 100, 3)
    assert (boxed.content_width, boxed.content_height) == (100, 50)
    assert boxed.offset_y == 25
    assert np.all(boxed.image[0] == (32, 32, 32))

    odd_geometry = np.zeros((4, 5, 3), dtype=np.uint8)
    fitted, scale = resize_to_fit(odd_geometry, 4, 2, allow_upscale=True)
    assert scale == pytest.approx(0.5)
    assert fitted.shape == (2, 3, 3)


def test_invalid_geometry() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be positive"):
        resize_exact(image, 0, 10, cv2.INTER_AREA)
    with pytest.raises(ValueError, match="must be positive"):
        resize_by_scale(image, -1.0, 1.0, cv2.INTER_LINEAR)
    with pytest.raises(ValueError, match="non-empty"):
        resize_by_scale(
            np.empty((0, 0, 3), dtype=np.uint8), 1.0, 1.0, cv2.INTER_LINEAR
        )
    with pytest.raises(ValueError, match="non-empty"):
        resize_to_fit(np.empty((0, 0, 3), dtype=np.uint8), 100, 100)


def test_cli_default_outputs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parent / "resize_image.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path / "outputs")],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "input=640x420 down=320x210 letterbox_content=640x420" in completed.stdout
    assert cv2.imread(str(tmp_path / "outputs" / "letterbox-640.png")).shape == (
        640,
        640,
        3,
    )
    assert DEFAULT_INPUT.is_file()

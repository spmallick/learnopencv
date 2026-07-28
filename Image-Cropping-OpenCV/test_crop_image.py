from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from crop_image import DEFAULT_INPUT, crop_image, extract_tiles, read_bgr


def test_crop_is_exact_independent_copy() -> None:
    image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    cropped = crop_image(image, 3, 2, 7, 5)
    assert np.array_equal(cropped, image[2:7, 3:10])
    cropped[:] = 0
    assert np.any(image[2:7, 3:10] != 0)


def test_roi_validation_rejects_out_of_bounds() -> None:
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds image bounds"):
        crop_image(image, 10, 2, 4, 5)
    with pytest.raises(ValueError, match="must be positive"):
        crop_image(image, 0, 0, 0, 5)


def test_partial_tiles_cover_every_pixel() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    tiles = extract_tiles(image, 6, 5)
    assert len(tiles) == 9
    assert sum(tile.image.shape[0] * tile.image.shape[1] for tile in tiles) == 12 * 16


def test_cli_default_outputs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parent / "crop_image.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path / "outputs")],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "crop=320x240 tiles=12" in completed.stdout
    assert cv2.imread(str(tmp_path / "outputs" / "cropped.png")).shape == (240, 320, 3)
    assert len(list((tmp_path / "outputs" / "patches").glob("*.png"))) == 12
    assert read_bgr(DEFAULT_INPUT).shape == (420, 640, 3)

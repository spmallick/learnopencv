from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from image_io import DEFAULT_INPUT, read_image, run_image_io, write_image


def test_lossless_and_lossy_round_trips(tmp_path: Path) -> None:
    metrics = run_image_io(DEFAULT_INPUT, tmp_path)
    assert metrics["width"] == 640
    assert metrics["height"] == 420
    assert metrics["color_channels"] == 3
    assert 0.0 < metrics["jpeg_mae"] < 15.0
    assert np.array_equal(
        read_image(DEFAULT_INPUT), read_image(tmp_path / "lossless-copy.png")
    )


def test_modes_and_errors(tmp_path: Path) -> None:
    assert read_image(DEFAULT_INPUT, "grayscale").ndim == 2
    assert read_image(DEFAULT_INPUT, "unchanged").shape[2] == 3
    with pytest.raises(ValueError, match="mode must be"):
        read_image(DEFAULT_INPUT, "rgb")
    with pytest.raises(FileNotFoundError, match="Could not decode"):
        read_image(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="empty image"):
        write_image(tmp_path / "empty.png", np.empty((0, 0), dtype=np.uint8))


def test_cli_runs_from_unrelated_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parent / "image_io.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path / "outputs")],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "size=640x420" in completed.stdout
    assert (tmp_path / "outputs" / "image-io-comparison.png").is_file()

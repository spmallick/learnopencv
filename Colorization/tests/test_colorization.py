from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2 as cv
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL = PROJECT_DIR / "models" / "colorization_eccv16.onnx"


pytestmark = pytest.mark.skipif(not MODEL.is_file(), reason="ONNX model is not present")


def test_image_cli(tmp_path: Path) -> None:
    output = tmp_path / "colorized.png"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "colorizeImage.py"),
            "--input",
            str(PROJECT_DIR / "greyscaleImage.png"),
            "--model",
            str(MODEL),
            "--output",
            str(output),
            "--no-display",
            "--validate",
        ],
        check=True,
        cwd=PROJECT_DIR,
    )
    image = cv.imread(str(output))
    source = cv.imread(str(PROJECT_DIR / "greyscaleImage.png"))
    assert image is not None
    assert source is not None
    assert image.shape == source.shape


def test_video_cli(tmp_path: Path) -> None:
    output = tmp_path / "colorized.avi"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "colorizeVideo.py"),
            "--input",
            str(PROJECT_DIR / "greyscaleVideo.mp4"),
            "--model",
            str(MODEL),
            "--output",
            str(output),
            "--max-frames",
            "2",
            "--no-display",
            "--validate",
        ],
        check=True,
        cwd=PROJECT_DIR,
    )
    capture = cv.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv.CAP_PROP_FRAME_COUNT)) == 2
    capture.release()

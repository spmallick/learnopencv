from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from videoRead import read_video
from videoWrite import transcode_video


def test_read_video_headlessly() -> None:
    summary = read_video(PROJECT_DIR / "chaplin.mp4", max_frames=8)
    assert summary["frames_read"] == 8
    assert summary["frame_size"] == [640, 360]
    assert summary["fps"] > 0
    assert summary["reported_frame_count"] >= 8
    # Decoder implementations may differ by a few values, so use a broad
    # luminance-content guard instead of pretending compressed pixels are exact.
    assert 80_000_000 < summary["first_frame_checksum"] < 100_000_000


def test_transcode_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "round-trip.avi"
    written = transcode_video(
        PROJECT_DIR / "chaplin.mp4",
        output,
        codec="MJPG",
        max_frames=8,
    )
    reread = read_video(output)

    assert written["frames_written"] == 8
    assert written["frame_size"] == [640, 360]
    assert output.stat().st_size > 10_000
    assert reread["frames_read"] == 8
    assert reread["frame_size"] == [640, 360]


def test_invalid_codec_and_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        transcode_video(
            PROJECT_DIR / "chaplin.mp4", tmp_path / "bad.avi", codec="BAD"
        )
    with pytest.raises(FileNotFoundError):
        read_video(tmp_path / "missing.mp4")
    with pytest.raises(ValueError):
        transcode_video(
            PROJECT_DIR / "chaplin.mp4",
            tmp_path / "bad-fps.avi",
            output_fps=math.nan,
        )

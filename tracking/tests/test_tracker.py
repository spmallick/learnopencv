from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tracker import _checked_bbox, create_tracker, parse_bbox, run_tracking


def test_bbox_parser() -> None:
    assert parse_bbox("10, 20, 30, 40") == (10, 20, 30, 40)
    with pytest.raises(Exception):
        parse_bbox("10,20,0,40")
    with pytest.raises(ValueError):
        _checked_bbox((0, 0, 0, 40), 640, 360)
    with pytest.raises(ValueError):
        _checked_bbox((600, 20, 50, 40), 640, 360)


def test_supported_tracker_is_available() -> None:
    tracker = create_tracker("MIL")
    assert tracker is not None
    with pytest.raises(ValueError):
        create_tracker("BOOSTING")


def test_headless_tracking_smoke(tmp_path: Path) -> None:
    output = tmp_path / "tracked.avi"
    snapshot = tmp_path / "tracked.png"
    summary = run_tracking(
        PROJECT_DIR / "videos" / "chaplin.mp4",
        (287, 23, 86, 320),
        tracker_name="MIL",
        output_path=output,
        snapshot_path=snapshot,
        max_frames=5,
    )

    assert summary["frames_processed"] == 5
    assert summary["successful_updates"] == 5
    assert summary["frame_size"] == [640, 360]
    assert len(summary["last_bbox"]) == 4
    assert output.stat().st_size > 1_000
    image = cv2.imread(str(snapshot))
    assert image is not None
    assert image.shape[:2] == (360, 640)


def test_missing_input_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_tracking(tmp_path / "missing.mp4", (0, 0, 10, 10), max_frames=1)

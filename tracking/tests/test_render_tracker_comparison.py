from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import render_tracker_comparison as renderer
from render_tracker_comparison import (
    SourceInfo,
    TrackResult,
    _install_output_pair,
    render_video,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _inject_json_install_failure(
    monkeypatch: pytest.MonkeyPatch,
    staged_json: Path,
    final_json: Path,
) -> None:
    real_replace = os.replace

    def failing_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        if Path(source) == staged_json and Path(destination) == final_json:
            raise OSError("injected JSON install failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", failing_replace)


def test_pair_install_replaces_both_outputs_and_sets_public_modes(
    tmp_path: Path,
) -> None:
    final_video = tmp_path / "result.mp4"
    final_json = tmp_path / "result.json"
    staged_video = tmp_path / ".new-video.mp4"
    staged_json = tmp_path / ".new-result.json"
    final_video.write_bytes(b"old-video")
    final_json.write_bytes(b"old-json")
    final_video.chmod(0o640)
    final_json.chmod(0o600)
    staged_video.write_bytes(b"new-video")
    staged_json.write_bytes(b"new-json")

    _install_output_pair(
        staged_video,
        final_video,
        staged_json,
        final_json,
        overwrite=True,
    )

    assert final_video.read_bytes() == b"new-video"
    assert final_json.read_bytes() == b"new-json"
    assert _mode(final_video) == 0o644
    assert _mode(final_json) == 0o644
    assert not staged_video.exists()
    assert not staged_json.exists()
    assert not any(
        "rollback-backup" in path.name for path in tmp_path.iterdir()
    )


def test_pair_install_restores_existing_pair_when_json_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_video = tmp_path / "result.mp4"
    final_json = tmp_path / "result.json"
    staged_video = tmp_path / ".new-video.mp4"
    staged_json = tmp_path / ".new-result.json"
    final_video.write_bytes(b"old-video")
    final_json.write_bytes(b"old-json")
    final_video.chmod(0o640)
    final_json.chmod(0o600)
    staged_video.write_bytes(b"new-video")
    staged_json.write_bytes(b"new-json")
    _inject_json_install_failure(monkeypatch, staged_json, final_json)

    with pytest.raises(OSError, match="injected JSON install failure"):
        _install_output_pair(
            staged_video,
            final_video,
            staged_json,
            final_json,
            overwrite=True,
        )

    assert final_video.read_bytes() == b"old-video"
    assert final_json.read_bytes() == b"old-json"
    assert _mode(final_video) == 0o640
    assert _mode(final_json) == 0o600
    assert not staged_video.exists()
    assert not staged_json.exists()
    assert not any(
        "rollback-backup" in path.name for path in tmp_path.iterdir()
    )


def test_pair_install_removes_partial_new_output_when_no_pair_existed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_video = tmp_path / "result.mp4"
    final_json = tmp_path / "result.json"
    staged_video = tmp_path / ".new-video.mp4"
    staged_json = tmp_path / ".new-result.json"
    staged_video.write_bytes(b"new-video")
    staged_json.write_bytes(b"new-json")
    _inject_json_install_failure(monkeypatch, staged_json, final_json)

    with pytest.raises(OSError, match="injected JSON install failure"):
        _install_output_pair(
            staged_video,
            final_video,
            staged_json,
            final_json,
            overwrite=True,
        )

    assert not final_video.exists()
    assert not final_json.exists()
    assert not staged_video.exists()
    assert not staged_json.exists()


def test_together_only_emits_one_grid_frame_per_selected_source_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SourceInfo(
        path=tmp_path / "source.mp4",
        width=128,
        height=72,
        fps=30.0,
        reported_frame_count=2,
        start_frame=10,
        end_frame=11,
        selected_frame_count=2,
        frame_hashes=("first", "second"),
        selected_sequence_sha256="sequence",
        file_sha256="source",
    )
    frames = [
        np.zeros((72, 128, 3), dtype=np.uint8),
        np.full((72, 128, 3), 32, dtype=np.uint8),
    ]
    results = [
        TrackResult(
            name=name,
            entries=[
                {
                    "source_frame": 10,
                    "phase": "initialization",
                    "box_returned": None,
                    "bbox_xywh": [10.0, 10.0, 20.0, 20.0],
                    "bbox_off_frame": False,
                },
                {
                    "source_frame": 11,
                    "phase": "update",
                    "box_returned": True,
                    "bbox_xywh": [11.0, 10.0, 20.0, 20.0],
                    "bbox_off_frame": False,
                },
            ],
            box_returned_updates=1,
            update_count=1,
            mean_tracking_ms=1.0,
        )
        for name in ("MIL", "DASIAMRPN", "NANO", "VIT")
    ]

    def fake_selected_frames(requested_source: SourceInfo):
        assert requested_source is source
        for offset, frame in enumerate(frames):
            yield offset, source.start_frame + offset, frame

    writers = []

    class FakeWriter:
        def __init__(
            self,
            output_path: Path,
            fps: float,
            *,
            crf: int,
            preset: str,
        ) -> None:
            assert fps == source.fps
            assert crf == 20
            assert preset == "medium"
            self.output_path = output_path
            self.frames_written = 0
            self.frames: list[np.ndarray] = []
            writers.append(self)

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(frame.copy())
            self.frames_written += 1

        def finish(self) -> Path:
            staged = self.output_path.with_name(".staged.mp4")
            staged.write_bytes(b"video")
            return staged

        def abort(self) -> None:
            pass

    monkeypatch.setattr(renderer, "selected_frames", fake_selected_frames)
    monkeypatch.setattr(renderer, "H264Writer", FakeWriter)

    staged, sections, frame_count = render_video(
        source,
        results,
        tmp_path / "comparison.mp4",
        "Limitations",
        title_seconds=5.0,
        crf=20,
        preset="medium",
        together_only=True,
    )

    assert staged.read_bytes() == b"video"
    assert frame_count == source.selected_frame_count == 2
    assert sections == [
        {
            "kind": "together",
            "trackers": ["MIL", "DASIAMRPN", "NANO", "VIT"],
            "output_start_frame": 0,
            "output_end_frame": 1,
            "source_start_frame": 10,
            "source_end_frame": 11,
        }
    ]
    assert len(writers) == 1
    assert len(writers[0].frames) == source.selected_frame_count

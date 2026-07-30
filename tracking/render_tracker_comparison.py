#!/usr/bin/env python3
"""Render a reproducible OpenCV tracker comparison as a publishable MP4.

The selected source-frame range is inclusive. Every tracker is initialized on
the same first frame and bounding box. By default, the output first shows the
trackers together, then replays the same source frames once per tracker. Pass
``--together-only`` to emit only the simultaneous comparison grid.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Sequence

import cv2
import numpy as np

from tracker import (
    DEFAULT_MODELS_DIR,
    MODEL_TRACKER_FILES,
    create_tracker,
    parse_bbox,
)


OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
SUPPORTED_TRACKERS = ("MIL", "DASIAMRPN", "NANO", "VIT")
TRACKER_ALIASES = {
    "MIL": "MIL",
    "DASIAMRPN": "DASIAMRPN",
    "DA_SIAM_RPN": "DASIAMRPN",
    "NANO": "NANO",
    "NANOTRACK": "NANO",
    "NANOTRACKV2": "NANO",
    "VIT": "VIT",
    "VITTRACK": "VIT",
}
DISPLAY_NAMES = {
    "MIL": "MIL",
    "DASIAMRPN": "DaSiamRPN",
    "NANO": "NanoTrack",
    "VIT": "VitTrack",
}
# BGR values used for both the panel accent and the tracked bounding box.
TRACKER_COLORS = {
    "MIL": (235, 178, 52),
    "DASIAMRPN": (45, 133, 245),
    "NANO": (94, 204, 92),
    "VIT": (216, 105, 196),
}
BACKGROUND = (22, 27, 34)
PANEL_BACKGROUND = (11, 16, 22)
TEXT = (244, 246, 248)
MUTED_TEXT = (174, 183, 194)


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    width: int
    height: int
    fps: float
    reported_frame_count: int | None
    start_frame: int
    end_frame: int
    selected_frame_count: int
    frame_hashes: tuple[str, ...]
    selected_sequence_sha256: str
    file_sha256: str


@dataclass
class TrackResult:
    name: str
    entries: list[dict[str, Any]]
    box_returned_updates: int
    update_count: int
    mean_tracking_ms: float | None


def parse_trackers(value: str) -> tuple[str, ...]:
    """Parse a comma-separated tracker list and normalize documented aliases."""
    raw_names = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_names:
        raise argparse.ArgumentTypeError("Provide at least one tracker")

    normalized: list[str] = []
    for raw_name in raw_names:
        key = raw_name.upper().replace("-", "").replace(" ", "")
        name = TRACKER_ALIASES.get(key)
        if name is None:
            choices = ", ".join(SUPPORTED_TRACKERS)
            raise argparse.ArgumentTypeError(
                f"Unsupported tracker '{raw_name}'. Choose from: {choices}"
            )
        if name in normalized:
            raise argparse.ArgumentTypeError(
                f"Tracker '{DISPLAY_NAMES[name]}' was listed more than once"
            )
        normalized.append(name)

    if len(normalized) > 4:
        raise argparse.ArgumentTypeError("At most four trackers can be compared")
    return tuple(normalized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file_provenance(
    tracker_names: Sequence[str], models_dir: Path
) -> list[dict[str, Any]]:
    """Fingerprint every ONNX file used by the requested tracker set."""
    records: list[dict[str, Any]] = []
    for tracker_name in tracker_names:
        for filename in MODEL_TRACKER_FILES.get(tracker_name, ()):
            path = models_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{DISPLAY_NAMES[tracker_name]} model is missing: {path}"
                )
            records.append(
                {
                    "tracker": tracker_name,
                    "filename": filename,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return records


def _frame_digest(frame: np.ndarray) -> str:
    return hashlib.blake2b(memoryview(frame), digest_size=16).hexdigest()


def _valid_fps(value: float) -> bool:
    return math.isfinite(value) and value > 0


def inspect_source(
    input_path: Path,
    start_frame: int,
    end_frame: int | None,
    bbox: tuple[int, int, int, int],
) -> SourceInfo:
    """Decode and fingerprint the exact selected source-frame sequence."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if start_frame < 0:
        raise ValueError("--start-frame must be zero or greater")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("--end-frame must be greater than or equal to --start-frame")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the input video: {input_path}")

    frame_hashes: list[str] = []
    sequence_digest = hashlib.sha256()
    width = height = 0
    last_decoded = -1
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not _valid_fps(fps):
            raise RuntimeError(
                f"The input video reports an invalid frame rate ({fps!r})"
            )
        reported_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_frame_count = (
            int(round(reported_count_value))
            if math.isfinite(reported_count_value) and reported_count_value > 0
            else None
        )

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            last_decoded = frame_index

            current_height, current_width = frame.shape[:2]
            if width == 0:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise RuntimeError(
                    "The source frame size changes inside the selected video"
                )

            if frame_index >= start_frame and (
                end_frame is None or frame_index <= end_frame
            ):
                digest = _frame_digest(frame)
                frame_hashes.append(digest)
                sequence_digest.update(bytes.fromhex(digest))

            if end_frame is not None and frame_index >= end_frame:
                break
            frame_index += 1
    finally:
        capture.release()

    if last_decoded < start_frame or not frame_hashes:
        raise RuntimeError(
            f"The source ended at frame {last_decoded}; frame {start_frame} "
            "could not be decoded"
        )
    if end_frame is not None and last_decoded < end_frame:
        raise RuntimeError(
            f"The requested inclusive end frame is {end_frame}, but decoding "
            f"stopped at frame {last_decoded}"
        )

    actual_end = start_frame + len(frame_hashes) - 1
    x, y, box_width, box_height = bbox
    if (
        x < 0
        or y < 0
        or box_width <= 0
        or box_height <= 0
        or x + box_width > width
        or y + box_height > height
    ):
        raise ValueError(
            f"Initial bounding box {bbox} lies outside the {width}x{height} "
            f"frame at source frame {start_frame}"
        )

    return SourceInfo(
        path=input_path,
        width=width,
        height=height,
        fps=fps,
        reported_frame_count=reported_frame_count,
        start_frame=start_frame,
        end_frame=actual_end,
        selected_frame_count=len(frame_hashes),
        frame_hashes=tuple(frame_hashes),
        selected_sequence_sha256=sequence_digest.hexdigest(),
        file_sha256=_sha256_file(input_path),
    )


def selected_frames(source: SourceInfo) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield the fingerprint-verified selected frames from a fresh decoder."""
    capture = cv2.VideoCapture(str(source.path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not reopen input video: {source.path}")

    selected_offset = 0
    try:
        for frame_index in range(source.end_frame + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Decoding failed at source frame {frame_index}; expected "
                    f"frames through {source.end_frame}"
                )
            if frame_index < source.start_frame:
                continue

            if frame.shape[:2] != (source.height, source.width):
                raise RuntimeError(
                    f"Frame {frame_index} changed size during a repeated decode"
                )
            digest = _frame_digest(frame)
            expected = source.frame_hashes[selected_offset]
            if digest != expected:
                raise RuntimeError(
                    f"Frame {frame_index} decoded inconsistently across passes"
                )
            yield selected_offset, frame_index, frame
            selected_offset += 1
    finally:
        capture.release()

    if selected_offset != source.selected_frame_count:
        raise RuntimeError(
            f"Decoded {selected_offset} selected frames; expected "
            f"{source.selected_frame_count}"
        )


def _finite_bbox(values: Sequence[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise RuntimeError(f"Tracker returned a {len(values)}-value bounding box")
    bbox = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in bbox):
        raise RuntimeError(f"Tracker returned a non-finite bounding box: {bbox}")
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise RuntimeError(f"Tracker returned a non-positive bounding box: {bbox}")
    return bbox


def _tracking_score(tracker_instance: Any) -> float | None:
    getter = getattr(tracker_instance, "getTrackingScore", None)
    if getter is None:
        return None
    try:
        score = float(getter())
    except (AttributeError, TypeError, ValueError, cv2.error):
        return None
    return score if math.isfinite(score) else None


def _json_bbox(bbox: Sequence[float] | None) -> list[float] | None:
    if bbox is None:
        return None
    return [round(float(value), 3) for value in bbox]


def _bbox_outside_source(
    bbox: Sequence[float], frame_width: int, frame_height: int
) -> bool:
    x, y, width, height = bbox
    return (
        x < 0
        or y < 0
        or x + width > frame_width
        or y + height > frame_height
    )


def track_source(
    source: SourceInfo,
    tracker_name: str,
    bbox: tuple[int, int, int, int],
    models_dir: Path,
) -> TrackResult:
    """Track one target and retain a result for every selected source frame."""
    try:
        tracker_instance = create_tracker(tracker_name, models_dir)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as exc:
        raise RuntimeError(
            f"Could not create {DISPLAY_NAMES[tracker_name]}: {exc}"
        ) from exc

    entries: list[dict[str, Any]] = []
    box_returned_updates = 0
    update_count = 0
    elapsed_total = 0.0
    try:
        for offset, frame_index, frame in selected_frames(source):
            if offset == 0:
                try:
                    initialized = tracker_instance.init(frame, bbox)
                except cv2.error as exc:
                    raise RuntimeError(
                        f"{DISPLAY_NAMES[tracker_name]} initialization raised "
                        f"an OpenCV error at source frame {frame_index}: {exc}"
                    ) from exc
                if initialized is False:
                    raise RuntimeError(
                        f"{DISPLAY_NAMES[tracker_name]} initialization failed "
                        f"at source frame {frame_index}"
                    )
                initial_bbox = _finite_bbox(bbox)
                score = _tracking_score(tracker_instance)
                entries.append(
                    {
                        "source_frame": frame_index,
                        "phase": "initialization",
                        "box_returned": None,
                        "bbox_xywh": _json_bbox(initial_bbox),
                        "bbox_off_frame": False,
                        "tracking_score": (
                            round(score, 6) if score is not None else None
                        ),
                        "tracking_ms": None,
                    }
                )
                continue

            started = perf_counter()
            try:
                found, updated_bbox = tracker_instance.update(frame)
            except cv2.error as exc:
                raise RuntimeError(
                    f"{DISPLAY_NAMES[tracker_name]} update raised an OpenCV "
                    f"error at source frame {frame_index}: {exc}"
                ) from exc
            elapsed_ms = 1000.0 * (perf_counter() - started)
            elapsed_total += elapsed_ms
            update_count += 1

            current_bbox = _finite_bbox(updated_bbox) if found else None
            if found:
                box_returned_updates += 1
            score = _tracking_score(tracker_instance)
            entries.append(
                {
                    "source_frame": frame_index,
                    "phase": "update",
                    "box_returned": bool(found),
                    "bbox_xywh": _json_bbox(current_bbox),
                    "bbox_off_frame": (
                        _bbox_outside_source(
                            current_bbox, source.width, source.height
                        )
                        if current_bbox is not None
                        else None
                    ),
                    "tracking_score": (
                        round(score, 6) if score is not None else None
                    ),
                    "tracking_ms": round(elapsed_ms, 3),
                }
            )
    finally:
        del tracker_instance
        gc.collect()

    if len(entries) != source.selected_frame_count:
        raise RuntimeError(
            f"{DISPLAY_NAMES[tracker_name]} produced {len(entries)} results; "
            f"expected {source.selected_frame_count}"
        )
    return TrackResult(
        name=tracker_name,
        entries=entries,
        box_returned_updates=box_returned_updates,
        update_count=update_count,
        mean_tracking_ms=(
            elapsed_total / update_count if update_count > 0 else None
        ),
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _contain_frame(
    canvas: np.ndarray,
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[float, int, int, int, int]:
    x, y, width, height = rect
    canvas[y : y + height, x : x + width] = PANEL_BACKGROUND
    frame_height, frame_width = frame.shape[:2]
    scale = min(width / frame_width, height / frame_height)
    render_width = max(1, int(round(frame_width * scale)))
    render_height = max(1, int(round(frame_height * scale)))
    resized = cv2.resize(
        frame,
        (render_width, render_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    render_x = x + (width - render_width) // 2
    render_y = y + (height - render_height) // 2
    canvas[
        render_y : render_y + render_height,
        render_x : render_x + render_width,
    ] = resized
    return scale, render_x, render_y, render_width, render_height


def _draw_panel(
    canvas: np.ndarray,
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    tracker_name: str,
    entry: dict[str, Any],
    *,
    compact: bool,
) -> None:
    x, y, width, height = rect
    color = TRACKER_COLORS[tracker_name]
    scale, render_x, render_y, render_width, render_height = _contain_frame(
        canvas, frame, rect
    )
    cv2.rectangle(
        canvas,
        (x + 1, y + 1),
        (x + width - 2, y + height - 2),
        color,
        2,
        cv2.LINE_AA,
    )

    bbox = entry["bbox_xywh"]
    box_available = entry["phase"] == "initialization" or entry["box_returned"]
    if box_available and bbox is not None:
        box_x, box_y, box_width, box_height = bbox
        left = int(round(render_x + box_x * scale))
        top = int(round(render_y + box_y * scale))
        right = int(round(render_x + (box_x + box_width) * scale))
        bottom = int(round(render_y + (box_y + box_height) * scale))
        image_left = render_x
        image_top = render_y
        image_right = render_x + render_width - 1
        image_bottom = render_y + render_height - 1
        left = max(image_left, min(image_right, left))
        right = max(image_left, min(image_right, right))
        top = max(image_top, min(image_bottom, top))
        bottom = max(image_top, min(image_bottom, bottom))
        if right > left and bottom > top:
            cv2.rectangle(
                canvas,
                (left, top),
                (right, bottom),
                color,
                4 if not compact else 3,
                cv2.LINE_AA,
            )

    header_height = 42 if compact else 54
    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (x + 2, y + 2),
        (x + width - 3, y + header_height),
        (5, 8, 12),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(
        canvas,
        (x + 2, y + 2),
        (x + 9, y + header_height),
        color,
        -1,
    )
    label_scale = 0.68 if compact else 0.9
    _put_text(
        canvas,
        DISPLAY_NAMES[tracker_name],
        (x + 20, y + (29 if compact else 37)),
        scale=label_scale,
        thickness=2,
    )
    # OpenCV's Boolean only says that an update returned a box; the box can
    # still drift to the wrong object, which the video intentionally exposes.
    if entry["phase"] == "initialization":
        status = "INITIAL ROI"
    elif not entry["box_returned"]:
        status = "NO BOX"
    elif entry["bbox_off_frame"]:
        status = "BOX OFF FRAME"
    else:
        status = "BOX RETURNED"
    status_scale = 0.43 if compact else 0.58
    status_size = cv2.getTextSize(
        status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, 1
    )[0]
    _put_text(
        canvas,
        status,
        (x + width - status_size[0] - 16, y + (28 if compact else 36)),
        scale=status_scale,
        color=(
            (106, 190, 255)
            if entry["bbox_off_frame"]
            else (
                (178, 225, 186)
                if box_available
                else (126, 126, 245)
            )
        ),
        thickness=1,
    )
    frame_label = f"Source frame {entry['source_frame']}"
    _put_text(
        canvas,
        frame_label,
        (x + 14, y + height - 13),
        scale=0.42 if compact else 0.55,
        color=MUTED_TEXT,
        thickness=1,
    )


def _grid_rectangles(count: int) -> tuple[tuple[int, int, int, int], ...]:
    if count == 1:
        return ((0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT),)
    if count == 2:
        return ((0, 180, 640, 360), (640, 180, 640, 360))
    if count == 3:
        return (
            (320, 0, 640, 360),
            (0, 360, 640, 360),
            (640, 360, 640, 360),
        )
    if count == 4:
        return (
            (0, 0, 640, 360),
            (640, 0, 640, 360),
            (0, 360, 640, 360),
            (640, 360, 640, 360),
        )
    raise ValueError("The comparison layout supports one to four trackers")


def _title_card(
    title: str,
    subtitle: str,
    tracker_names: Sequence[str],
) -> np.ndarray:
    canvas = np.full(
        (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), BACKGROUND, dtype=np.uint8
    )
    cv2.rectangle(canvas, (0, 0), (OUTPUT_WIDTH, 10), (52, 180, 235), -1)
    cv2.rectangle(
        canvas,
        (96, 196),
        (1184, 520),
        (29, 35, 43),
        -1,
        cv2.LINE_AA,
    )
    title_scale = 1.35 if len(title) < 38 else 1.05
    title_size = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 2
    )[0]
    _put_text(
        canvas,
        title,
        ((OUTPUT_WIDTH - title_size[0]) // 2, 310),
        scale=title_scale,
        thickness=2,
    )
    subtitle_scale = 0.65
    subtitle_size = cv2.getTextSize(
        subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, 1
    )[0]
    _put_text(
        canvas,
        subtitle,
        ((OUTPUT_WIDTH - subtitle_size[0]) // 2, 365),
        scale=subtitle_scale,
        color=MUTED_TEXT,
        thickness=1,
    )

    if tracker_names:
        total_width = sum(
            cv2.getTextSize(
                DISPLAY_NAMES[name],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )[0][0]
            + 48
            for name in tracker_names
        )
        cursor = (OUTPUT_WIDTH - total_width) // 2
        for tracker_name in tracker_names:
            color = TRACKER_COLORS[tracker_name]
            cv2.circle(canvas, (cursor + 9, 418), 7, color, -1, cv2.LINE_AA)
            _put_text(
                canvas,
                DISPLAY_NAMES[tracker_name],
                (cursor + 25, 424),
                scale=0.55,
                thickness=1,
            )
            cursor += (
                cv2.getTextSize(
                    DISPLAY_NAMES[tracker_name],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    1,
                )[0][0]
                + 48
            )
    return canvas


class H264Writer:
    """Stream BGR frames to ffmpeg and atomically finalize an H.264 MP4."""

    def __init__(
        self,
        output_path: Path,
        fps: float,
        *,
        crf: int,
        preset: str,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to create the H.264 MP4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".partial.mp4",
            dir=output_path.parent,
            delete=False,
        )
        temporary.close()
        self.output_path = output_path
        self.temporary_path = Path(temporary.name)
        self.frames_written = 0
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
            "-framerate",
            f"{fps:.12g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temporary_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3):
            raise ValueError(
                f"Output frame has shape {frame.shape}; expected "
                f"({OUTPUT_HEIGHT}, {OUTPUT_WIDTH}, 3)"
            )
        if frame.dtype != np.uint8:
            raise ValueError("Output frames must use uint8 pixels")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is unavailable")
        try:
            written = self.process.stdin.write(
                np.ascontiguousarray(frame).tobytes()
            )
        except BrokenPipeError as exc:
            stderr = (
                self.process.stderr.read().decode("utf-8", "replace")
                if self.process.stderr is not None
                else ""
            )
            raise RuntimeError(f"ffmpeg stopped while encoding: {stderr}") from exc
        expected = OUTPUT_WIDTH * OUTPUT_HEIGHT * 3
        if written != expected:
            raise RuntimeError(
                f"ffmpeg accepted {written} frame bytes; expected {expected}"
            )
        self.frames_written += 1

    def finish(self) -> Path:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", "replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ffmpeg failed with exit code {return_code}: {stderr.strip()}"
            )
        if not self.temporary_path.is_file() or self.temporary_path.stat().st_size == 0:
            self.temporary_path.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg reported success but produced no video")
        return self.temporary_path

    def abort(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
        self.temporary_path.unlink(missing_ok=True)


def render_video(
    source: SourceInfo,
    results: Sequence[TrackResult],
    output_path: Path,
    scene_title: str,
    title_seconds: float,
    crf: int,
    preset: str,
    together_only: bool = False,
) -> tuple[Path, list[dict[str, Any]], int]:
    """Encode the simultaneous grid, optionally followed by solo replays."""
    title_frame_count = (
        0 if together_only else int(round(title_seconds * source.fps))
    )
    if not together_only and title_seconds > 0 and title_frame_count < 1:
        title_frame_count = 1
    rectangles = _grid_rectangles(len(results))
    sections: list[dict[str, Any]] = []
    timeline_frame = 0
    writer = H264Writer(output_path, source.fps, crf=crf, preset=preset)

    def write_card(
        title: str, subtitle: str, names: Sequence[str], section_name: str
    ) -> None:
        nonlocal timeline_frame
        if title_frame_count == 0:
            return
        start = timeline_frame
        card = _title_card(title, subtitle, names)
        for _ in range(title_frame_count):
            writer.write(card)
            timeline_frame += 1
        sections.append(
            {
                "kind": "title",
                "name": section_name,
                "output_start_frame": start,
                "output_end_frame": timeline_frame - 1,
            }
        )

    try:
        all_names = [result.name for result in results]
        write_card(
            f"{scene_title}: Trackers Together",
            (
                f"Same source frames {source.start_frame}-{source.end_frame} "
                "| Same initial ROI"
            ),
            all_names,
            "together-title",
        )
        together_start = timeline_frame
        for offset, _, frame in selected_frames(source):
            canvas = np.full(
                (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), BACKGROUND, dtype=np.uint8
            )
            for rect, result in zip(rectangles, results):
                _draw_panel(
                    canvas,
                    frame,
                    rect,
                    result.name,
                    result.entries[offset],
                    compact=len(results) > 1,
                )
            writer.write(canvas)
            timeline_frame += 1
        sections.append(
            {
                "kind": "together",
                "trackers": all_names,
                "output_start_frame": together_start,
                "output_end_frame": timeline_frame - 1,
                "source_start_frame": source.start_frame,
                "source_end_frame": source.end_frame,
            }
        )

        if not together_only:
            for result in results:
                write_card(
                    f"{scene_title}: {DISPLAY_NAMES[result.name]}",
                    (
                        f"Source frames {source.start_frame}-{source.end_frame} "
                        "| Full-screen replay"
                    ),
                    [result.name],
                    f"{result.name.lower()}-title",
                )
                tracker_start = timeline_frame
                for offset, _, frame in selected_frames(source):
                    canvas = np.full(
                        (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3),
                        BACKGROUND,
                        dtype=np.uint8,
                    )
                    _draw_panel(
                        canvas,
                        frame,
                        (0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT),
                        result.name,
                        result.entries[offset],
                        compact=False,
                    )
                    writer.write(canvas)
                    timeline_frame += 1
                sections.append(
                    {
                        "kind": "single",
                        "tracker": result.name,
                        "output_start_frame": tracker_start,
                        "output_end_frame": timeline_frame - 1,
                        "source_start_frame": source.start_frame,
                        "source_end_frame": source.end_frame,
                    }
                )

        temporary_path = writer.finish()
    except BaseException:
        writer.abort()
        raise

    if writer.frames_written != timeline_frame:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Encoder received {writer.frames_written} frames, but the timeline "
            f"contains {timeline_frame}"
        )
    return temporary_path, sections, timeline_frame


def _top_level_mp4_atoms(path: Path) -> list[tuple[str, int]]:
    atoms: list[tuple[str, int]] = []
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset + 8 <= file_size:
            stream.seek(offset)
            header = stream.read(8)
            size, atom_type_bytes = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise RuntimeError("The encoded MP4 has a truncated atom header")
                size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                raise RuntimeError("The encoded MP4 has an invalid top-level atom")
            atom_type = atom_type_bytes.decode("ascii", "replace")
            atoms.append((atom_type, offset))
            offset += size
    return atoms


def verify_encoded_video(
    path: Path, expected_frames: int, expected_fps: float
) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to verify the encoded MP4")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe could not verify the MP4: {completed.stderr}")
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc
    streams = probe.get("streams", [])
    video_streams = [
        stream for stream in streams if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1 or audio_streams:
        raise RuntimeError(
            "The output must contain exactly one video stream and no audio"
        )
    video = video_streams[0]
    checks = {
        "codec_name": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
    }
    if checks != {
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "width": OUTPUT_WIDTH,
        "height": OUTPUT_HEIGHT,
    }:
        raise RuntimeError(f"Unexpected encoded stream properties: {checks}")

    decoded_frames_raw = video.get("nb_read_frames")
    if decoded_frames_raw in (None, "N/A"):
        raise RuntimeError("ffprobe did not report the decoded frame count")
    decoded_frames = int(decoded_frames_raw)
    if decoded_frames != expected_frames:
        raise RuntimeError(
            f"The output contains {decoded_frames} frames; expected "
            f"{expected_frames}"
        )

    rate_text = video.get("avg_frame_rate", "0/0")
    numerator_text, denominator_text = rate_text.split("/", maxsplit=1)
    denominator = float(denominator_text)
    encoded_fps = float(numerator_text) / denominator if denominator else 0.0
    if not math.isclose(encoded_fps, expected_fps, rel_tol=1e-5, abs_tol=1e-3):
        raise RuntimeError(
            f"The encoded frame rate is {encoded_fps}; expected {expected_fps}"
        )

    atoms = _top_level_mp4_atoms(path)
    atom_positions = {name: offset for name, offset in atoms}
    if "moov" not in atom_positions or "mdat" not in atom_positions:
        raise RuntimeError("The encoded MP4 is missing its moov or mdat atom")
    if atom_positions["moov"] > atom_positions["mdat"]:
        raise RuntimeError("The MP4 was not finalized for fast-start playback")

    return {
        **checks,
        "fps": encoded_fps,
        "decoded_frame_count": decoded_frames,
        "faststart": True,
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _stage_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write and sync JSON beside its destination without installing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.stem}.",
        suffix=".partial.json",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, 0o644)
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _install_output_pair(
    staged_video: Path,
    output_path: Path,
    staged_json: Path,
    json_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Install a staged MP4/JSON pair and restore the prior pair on failure."""
    staged_pairs = (
        (staged_video, output_path),
        (staged_json, json_path),
    )
    backups: list[tuple[Path, Path]] = []
    installed: list[Path] = []

    try:
        for staged, final in staged_pairs:
            if not staged.is_file():
                raise FileNotFoundError(f"Staged output is missing: {staged}")
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists() and not final.is_file():
                raise ValueError(f"Output path is not a regular file: {final}")
            if final.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {final} "
                    "(pass --overwrite to replace it)"
                )

        for _, final in staged_pairs:
            if not final.exists():
                continue
            backup_file = tempfile.NamedTemporaryFile(
                prefix=f".{final.name}.",
                suffix=".rollback-backup",
                dir=final.parent,
                delete=False,
            )
            backup_file.close()
            backup_path = Path(backup_file.name)
            try:
                os.replace(final, backup_path)
            except BaseException:
                backup_path.unlink(missing_ok=True)
                raise
            backups.append((final, backup_path))

        for staged, final in staged_pairs:
            os.replace(staged, final)
            installed.append(final)
            os.chmod(final, 0o644)
    except BaseException as install_error:
        backup_by_final = {final: backup for final, backup in backups}
        rollback_errors: list[str] = []

        # New outputs that had no predecessor must disappear on rollback.
        for final in reversed(installed):
            if final in backup_by_final:
                continue
            try:
                final.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"remove {final}: {exc}")

        # Replacing a new file with its backup is itself atomic.
        for final, backup in reversed(backups):
            try:
                if not backup.is_file():
                    raise FileNotFoundError(f"backup disappeared: {backup}")
                os.replace(backup, final)
            except OSError as exc:
                rollback_errors.append(
                    f"restore {final} from {backup}: {exc}"
                )

        for staged, _ in staged_pairs:
            try:
                staged.unlink(missing_ok=True)
            except OSError as exc:
                rollback_errors.append(f"remove staged file {staged}: {exc}")

        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                "Output-pair installation failed and rollback was incomplete: "
                f"{details}"
            ) from install_error
        raise

    cleanup_errors: list[str] = []
    for _, backup in backups:
        try:
            backup.unlink(missing_ok=True)
        except OSError as exc:
            cleanup_errors.append(f"{backup}: {exc}")
    if cleanup_errors:
        raise RuntimeError(
            "The new output pair is installed, but rollback-backup cleanup "
            f"failed: {'; '.join(cleanup_errors)}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    json_path = (
        args.json_output.expanduser().resolve()
        if args.json_output is not None
        else output_path.with_suffix(".json")
    )
    models_dir = args.models_dir.expanduser().resolve()

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("--output must use an .mp4 filename")
    if output_path == json_path:
        raise ValueError("The MP4 and JSON output paths must be different")
    for path in (output_path, json_path):
        if path == input_path or (
            path.exists() and input_path.exists() and os.path.samefile(path, input_path)
        ):
            raise ValueError(
                f"Refusing to overwrite the source video with an output: {path}"
            )
    for path in (output_path, json_path):
        if path.exists() and not path.is_file():
            raise ValueError(f"Output path is not a regular file: {path}")
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {path} (pass --overwrite to replace it)"
            )
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    if not math.isfinite(args.title_seconds) or args.title_seconds < 0:
        raise ValueError("--title-seconds must be zero or greater")

    source = inspect_source(
        input_path, args.start_frame, args.end_frame, args.bbox
    )
    model_files = model_file_provenance(args.trackers, models_dir)
    results = [
        track_source(source, name, args.bbox, models_dir)
        for name in args.trackers
    ]
    if model_file_provenance(args.trackers, models_dir) != model_files:
        raise RuntimeError("A tracker model file changed while tracking was running")
    scene_title = args.scene_title or input_path.stem.replace("_", " ").title()
    temporary_video, sections, output_frame_count = render_video(
        source,
        results,
        output_path,
        scene_title,
        args.title_seconds,
        args.crf,
        args.preset,
        together_only=args.together_only,
    )
    staged_json: Path | None = None
    try:
        verification = verify_encoded_video(
            temporary_video, output_frame_count, source.fps
        )
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "opencv_version": cv2.__version__,
            "source": {
                "filename": source.path.name,
                "file_sha256": source.file_sha256,
                "width": source.width,
                "height": source.height,
                "fps": source.fps,
                "reported_frame_count": source.reported_frame_count,
                "selected_start_frame_inclusive": source.start_frame,
                "selected_end_frame_inclusive": source.end_frame,
                "selected_frame_count": source.selected_frame_count,
                "selected_sequence_sha256": source.selected_sequence_sha256,
                "initial_bbox_xywh": list(args.bbox),
            },
            "models": {
                "files": model_files,
            },
            "trackers": [
                {
                    "name": result.name,
                    "display_name": DISPLAY_NAMES[result.name],
                    "color_rgb": list(reversed(TRACKER_COLORS[result.name])),
                    "update_count": result.update_count,
                    "box_returned_updates": result.box_returned_updates,
                    "box_return_rate": (
                        result.box_returned_updates / result.update_count
                        if result.update_count
                        else None
                    ),
                    "mean_tracking_ms": (
                        round(result.mean_tracking_ms, 3)
                        if result.mean_tracking_ms is not None
                        else None
                    ),
                    "trajectory": result.entries,
                }
                for result in results
            ],
            "output": {
                "filename": output_path.name,
                "scene_title": scene_title,
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "fps": source.fps,
                "frame_count": output_frame_count,
                "title_seconds": (
                    0.0 if args.together_only else args.title_seconds
                ),
                "layout": (
                    "together-only"
                    if args.together_only
                    else "together-and-sequential"
                ),
                "sections": sections,
                "verification": verification,
            },
        }
        os.chmod(temporary_video, 0o644)
        staged_json = _stage_json(json_path, metadata)
        _install_output_pair(
            temporary_video,
            output_path,
            staged_json,
            json_path,
            overwrite=args.overwrite,
        )
    except BaseException:
        temporary_video.unlink(missing_ok=True)
        if staged_json is not None:
            staged_json.unlink(missing_ok=True)
        raise

    return {
        "video": str(output_path),
        "json": str(json_path),
        "trackers": [DISPLAY_NAMES[result.name] for result in results],
        "source_frames": [
            source.start_frame,
            source.end_frame,
        ],
        "output_frames": output_frame_count,
        "verification": verification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="source video")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="final 1280x720 H.264 MP4",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="trajectory/result JSON (defaults beside the MP4)",
    )
    parser.add_argument(
        "--trackers",
        type=parse_trackers,
        default=parse_trackers("MIL,DASIAMRPN,NANO,VIT"),
        help=(
            "comma-separated list using MIL, DaSiamRPN, NanoTrack, and/or "
            "VitTrack"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="directory containing the verified ONNX tracker models",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        required=True,
        help="initial x,y,width,height on --start-frame",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="first source frame to include (zero-based and inclusive)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="last source frame to include (inclusive); omit to use the rest",
    )
    parser.add_argument(
        "--scene-title",
        help="short title shown on section cards",
    )
    parser.add_argument(
        "--title-seconds",
        type=float,
        default=1.0,
        help=(
            "duration of each title card; use 0 to disable "
            "(ignored with --together-only)"
        ),
    )
    parser.add_argument(
        "--together-only",
        action="store_true",
        help=(
            "emit only the simultaneous comparison grid, with no title cards "
            "or full-screen replays"
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="libx264 quality setting (lower is higher quality)",
    )
    parser.add_argument(
        "--preset",
        choices=(
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
        ),
        default="medium",
        help="libx264 encoding preset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing MP4/JSON outputs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        cv2.error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

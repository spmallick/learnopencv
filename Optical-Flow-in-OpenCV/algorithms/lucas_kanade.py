"""Sparse pyramidal Lucas-Kanade tracking for the optical-flow tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class SparseFlowSummary:
    """Stable metrics returned to the CLI and regression tests."""

    frame_pairs: int
    mean_magnitude: float
    output_path: Path


def run_lucas_kanade(
    *,
    video_path: Path,
    output_dir: Path,
    show_windows: bool,
    max_frames: int,
    validate: bool,
) -> SparseFlowSummary:
    """Track good features across a video and save the final trail image."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {video_path}")

    has_frame, previous_bgr = capture.read()
    if not has_frame or previous_bgr is None:
        capture.release()
        raise RuntimeError(f"Input video has no readable frames: {video_path}")

    previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
    previous_points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )
    if previous_points is None:
        capture.release()
        raise RuntimeError("No Shi-Tomasi features were found in the first frame")

    generator = np.random.default_rng(7)
    colors = generator.integers(0, 256, size=(100, 3), dtype=np.uint8)
    trails = np.zeros_like(previous_bgr)
    frame_pairs = 0
    magnitudes: list[float] = []
    visualization = previous_bgr.copy()

    while max_frames == 0 or frame_pairs < max_frames:
        has_frame, current_bgr = capture.read()
        if not has_frame or current_bgr is None:
            break

        current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03,
            ),
        )
        if current_points is None or status is None:
            break

        valid = status.reshape(-1) == 1
        good_new = current_points[valid]
        good_old = previous_points[valid]
        if len(good_new) == 0:
            break

        displacement = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
        magnitudes.append(float(np.mean(np.linalg.norm(displacement, axis=1))))

        frame = current_bgr.copy()
        for index, (new, old) in enumerate(zip(good_new, good_old)):
            new_xy = tuple(np.rint(new.ravel()).astype(int))
            old_xy = tuple(np.rint(old.ravel()).astype(int))
            color = colors[index % len(colors)].tolist()
            cv2.line(trails, new_xy, old_xy, color, 2, cv2.LINE_AA)
            cv2.circle(frame, new_xy, 4, color, -1, cv2.LINE_AA)

        visualization = cv2.add(frame, trails)
        frame_pairs += 1

        if show_windows:
            cv2.imshow("Sparse Lucas-Kanade optical flow", visualization)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                trails.fill(0)

        previous_gray = current_gray
        previous_points = good_new.reshape(-1, 1, 2)

    capture.release()
    if show_windows:
        cv2.destroyAllWindows()

    if frame_pairs == 0:
        raise RuntimeError("No feature tracks were produced from the video")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lucaskanade-optical-flow.png"
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Unable to write optical-flow image: {output_path}")

    overall_mean = float(np.mean(magnitudes))
    if validate:
        if not math.isfinite(overall_mean) or overall_mean <= 0.0:
            raise RuntimeError(
                f"Expected finite nonzero tracked motion, got {overall_mean}"
            )
        reloaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if reloaded is None or reloaded.size == 0:
            raise RuntimeError("Saved sparse-flow visualization is unreadable")

    return SparseFlowSummary(frame_pairs, overall_mean, output_path)

"""Dense optical-flow implementations and visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class DenseFlowSummary:
    """Stable metrics returned to the CLI and regression tests."""

    frame_pairs: int
    mean_magnitude: float
    output_path: Path


def flow_to_bgr(flow: np.ndarray) -> tuple[np.ndarray, float]:
    """Encode horizontal/vertical flow as hue/value and return mean magnitude."""

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * 90.0 / np.pi, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), float(np.mean(magnitude))


def calculate_dense_flow(
    algorithm: str,
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
) -> np.ndarray:
    """Call one modern API available in both supported OpenCV majors."""

    if algorithm == "farneback":
        previous = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
        current = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            f"{algorithm} requires opencv-contrib-python, not opencv-python"
        )

    if algorithm == "lucaskanade_dense":
        previous = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
        current = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.optflow.calcOpticalFlowSparseToDense(
            previous,
            current,
            None,
        )
    if algorithm == "rlof":
        return cv2.optflow.calcOpticalFlowDenseRLOF(
            previous_bgr,
            current_bgr,
            None,
        )
    raise ValueError(f"Unsupported dense optical-flow algorithm: {algorithm}")


def run_dense_optical_flow(
    *,
    algorithm: str,
    video_path: Path,
    output_dir: Path,
    show_windows: bool,
    max_frames: int,
    validate: bool,
) -> DenseFlowSummary:
    """Process a video without assuming the caller's working directory."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {video_path}")

    has_frame, previous_bgr = capture.read()
    if not has_frame or previous_bgr is None:
        capture.release()
        raise RuntimeError(f"Input video has no readable frames: {video_path}")

    frame_pairs = 0
    magnitudes: list[float] = []
    visualization = None

    while max_frames == 0 or frame_pairs < max_frames:
        has_frame, current_bgr = capture.read()
        if not has_frame or current_bgr is None:
            break

        flow = calculate_dense_flow(algorithm, previous_bgr, current_bgr)
        visualization, mean_magnitude = flow_to_bgr(flow)
        magnitudes.append(mean_magnitude)
        frame_pairs += 1

        if show_windows:
            cv2.imshow("Input frame", current_bgr)
            cv2.imshow("Dense optical flow", visualization)
            key = cv2.waitKey(25) & 0xFF
            if key in (27, ord("q")):
                break

        previous_bgr = current_bgr

    capture.release()
    if show_windows:
        cv2.destroyAllWindows()

    if visualization is None or frame_pairs == 0:
        raise RuntimeError("No frame pairs were available for optical flow")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{algorithm}-optical-flow.png"
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Unable to write optical-flow image: {output_path}")

    overall_mean = float(np.mean(magnitudes))
    if validate:
        if not math.isfinite(overall_mean) or overall_mean <= 0.0:
            raise RuntimeError(
                f"Expected finite nonzero motion, got {overall_mean}"
            )
        reloaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if reloaded is None or reloaded.size == 0:
            raise RuntimeError("Saved optical-flow visualization is unreadable")

    return DenseFlowSummary(frame_pairs, overall_mean, output_path)

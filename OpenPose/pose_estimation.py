"""Shared MediaPipe Pose inference helpers for the image and video examples.

The model is the Apache-2.0 MediaPipe Pose ONNX export published by OpenCV Zoo.
It estimates one person at a time, so these helpers deliberately treat the
complete input frame as the person's region of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# Resolve bundled defaults from this file so callers may run the examples from
# any working directory.
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "pose_estimation_mediapipe_2023mar.onnx"

# MediaPipe Pose defines 33 public landmarks. The six additional model outputs
# are auxiliary points used while refining the person crop, so they are not
# drawn as part of the public skeleton.
LANDMARK_COUNT = 33
MODEL_LANDMARK_COUNT = 39
MODEL_INPUT_SIZE = 256

# Each pair follows the official MediaPipe landmark numbering.
POSE_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)


@dataclass(frozen=True)
class SquareTransform:
    """Mapping from the model's padded square back to the source frame."""

    side: int
    left: int
    top: int


@dataclass(frozen=True)
class PoseResult:
    """Decoded screen-space landmarks and the model's pose confidence."""

    landmarks: np.ndarray
    confidence: float

    def visible_mask(self, threshold: float) -> np.ndarray:
        """Return landmarks that are both visible and present."""

        return (self.landmarks[:, 3] >= threshold) & (
            self.landmarks[:, 4] >= threshold
        )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert model logits to probabilities without numerical overflow."""

    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def configure_backend(net: cv2.dnn.Net, device: str) -> None:
    """Select a matching DNN backend and target for CPU or CUDA inference."""

    if device == "cpu":
        # DNN_BACKEND_DEFAULT selects the compatible graph engine in both
        # OpenCV 4.14 and 5.0; the target call explicitly keeps execution on CPU.
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return
    if device == "cuda":
        # CUDA works only when the local OpenCV build includes that backend.
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return
    raise ValueError(f"Unsupported device '{device}'. Choose 'cpu' or 'cuda'.")


def load_pose_model(model_path: Path, device: str = "cpu") -> cv2.dnn.Net:
    """Load the checked ONNX model and configure its execution device."""

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}. Run download_models.py first."
        )
    net = cv2.dnn.readNet(str(model_path))
    if net.empty():
        raise RuntimeError(f"OpenCV could not load the pose model: {model_path}")
    configure_backend(net, device)
    return net


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, SquareTransform]:
    """Letterbox a BGR frame to a 256x256 NHWC RGB float tensor."""

    if frame is None or frame.size == 0:
        raise ValueError("Cannot preprocess an empty frame.")
    height, width = frame.shape[:2]
    side = max(height, width)
    left = (side - width) // 2
    right = side - width - left
    top = (side - height) // 2
    bottom = side - height - top

    # Black letterboxing preserves the full person without changing aspect ratio.
    square = cv2.copyMakeBorder(
        frame,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    resized = cv2.resize(
        square,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0

    # This OpenCV Zoo export consumes channels-last (NHWC), not blobFromImage's
    # usual channels-first layout.
    blob = normalized[np.newaxis, ...]
    return np.ascontiguousarray(blob), SquareTransform(side, left, top)


def _find_output(outputs: Iterable[np.ndarray], element_count: int) -> np.ndarray:
    """Find a uniquely sized model output independent of output-layer ordering."""

    matches = [output for output in outputs if output.size == element_count]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one pose output with {element_count} values, "
            f"found {len(matches)}."
        )
    return matches[0]


def decode_pose(
    outputs: Iterable[np.ndarray],
    transform: SquareTransform,
) -> PoseResult:
    """Decode the 39x5 model tensor into 33 source-image landmarks."""

    output_list = list(outputs)
    raw_landmarks = _find_output(
        output_list, MODEL_LANDMARK_COUNT * 5
    ).reshape(MODEL_LANDMARK_COUNT, 5)
    raw_confidence = _find_output(output_list, 1).reshape(-1)

    # Work on a copy so callers may retain the raw DNN outputs for diagnostics.
    landmarks = raw_landmarks[:LANDMARK_COUNT].astype(np.float32, copy=True)
    scale = float(transform.side) / float(MODEL_INPUT_SIZE)
    landmarks[:, 0] = landmarks[:, 0] * scale - transform.left
    landmarks[:, 1] = landmarks[:, 1] * scale - transform.top
    landmarks[:, 2] *= scale
    landmarks[:, 3:5] = _sigmoid(landmarks[:, 3:5])

    return PoseResult(landmarks=landmarks, confidence=float(raw_confidence[0]))


def infer_pose(
    net: cv2.dnn.Net,
    frame: np.ndarray,
) -> PoseResult:
    """Run single-person pose estimation for one BGR image."""

    blob, transform = preprocess(frame)
    net.setInput(blob)
    output_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_names)
    return decode_pose(outputs, transform)


def draw_pose(
    frame: np.ndarray,
    result: PoseResult,
    score_threshold: float = 0.5,
) -> tuple[np.ndarray, int, int]:
    """Draw visible landmarks and return image, point count, and edge count."""

    output = frame.copy()
    keep = result.visible_mask(score_threshold)
    points = np.rint(result.landmarks[:, :2]).astype(np.int32)
    height, width = frame.shape[:2]

    # A point outside the image cannot be drawn safely even when the network
    # predicts high visibility.
    in_bounds = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    keep &= in_bounds

    edge_count = 0
    for start, end in POSE_EDGES:
        if keep[start] and keep[end]:
            cv2.line(
                output,
                tuple(points[start]),
                tuple(points[end]),
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            edge_count += 1

    for index, point in enumerate(points):
        if keep[index]:
            cv2.circle(
                output,
                tuple(point),
                4,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

    cv2.putText(
        output,
        f"Pose confidence: {result.confidence:.3f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output, int(np.count_nonzero(keep)), edge_count


def validate_pose(
    frame: np.ndarray,
    result: PoseResult,
    visible_count: int,
    edge_count: int,
) -> None:
    """Check stable structural invariants without pinning version-sensitive pixels."""

    if result.landmarks.shape != (LANDMARK_COUNT, 5):
        raise RuntimeError(
            f"Expected {(LANDMARK_COUNT, 5)} landmarks, "
            f"got {result.landmarks.shape}."
        )
    if not np.isfinite(result.landmarks).all():
        raise RuntimeError("Pose output contains NaN or infinite values.")
    if not np.isfinite(result.confidence) or not 0.0 <= result.confidence <= 1.0:
        raise RuntimeError(f"Invalid pose confidence: {result.confidence}")
    if frame is None or frame.size == 0:
        raise RuntimeError("Validation received an empty source frame.")
    if not 0 <= visible_count <= LANDMARK_COUNT:
        raise RuntimeError(f"Invalid visible landmark count: {visible_count}")
    if not 0 <= edge_count <= len(POSE_EDGES):
        raise RuntimeError(f"Invalid skeleton edge count: {edge_count}")


def write_image(path: Path, image: np.ndarray) -> None:
    """Create the output directory and fail clearly when encoding fails."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write output image: {path}")

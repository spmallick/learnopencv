"""Shared detectors, drawing, validation, and output helpers.

YuNet is the current OpenCV 4/5 path. Haar and dlib HOG are available only as
explicit historical baselines when their local dependencies exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_YUNET_MODEL = PROJECT_DIR / "models" / "face_detection_yunet_2026may.onnx"
DEFAULT_HAAR_MODEL = PROJECT_DIR / "models" / "haarcascade_frontalface_default.xml"

# YuNet returns five landmark pairs after its x, y, width, and height fields.
LANDMARK_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
)


class Detector(Protocol):
    """Small common interface used by the single and comparison CLIs."""

    name: str

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return rows shaped as x, y, width, height, landmarks, score."""


def backend_target(device: str) -> tuple[int, int]:
    """Return a correctly paired DNN backend and target."""

    if device == "cpu":
        return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU
    if device == "cuda":
        return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    raise ValueError(f"Unsupported device '{device}'. Choose 'cpu' or 'cuda'.")


class YuNetDetector:
    """Official OpenCV FaceDetectorYN wrapper for the dynamic YuNet model."""

    name = "YuNet"

    def __init__(
        self,
        model_path: Path = DEFAULT_YUNET_MODEL,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        device: str = "cpu",
    ) -> None:
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}. "
                "Run download_models.py first."
            )
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("YuNet score threshold must be between 0 and 1.")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("YuNet NMS threshold must be between 0 and 1.")
        if top_k <= 0:
            raise ValueError("YuNet top-k must be positive.")

        backend_id, target_id = backend_target(device)
        # The dynamic model accepts changing image dimensions. We still provide
        # a valid creation size, then set each real frame size before detect().
        self._model = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
            backend_id,
            target_id,
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces and normalize OpenCV's no-result value to an empty array."""

        if frame is None or frame.size == 0:
            raise ValueError("Cannot run YuNet on an empty frame.")
        height, width = frame.shape[:2]
        self._model.setInputSize((width, height))
        _, faces = self._model.detect(frame)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        return np.asarray(faces, dtype=np.float32).reshape(-1, 15)


class HaarDetector:
    """Optional OpenCV Haar baseline retained for historical comparison."""

    name = "Haar"

    def __init__(
        self,
        cascade_path: Path = DEFAULT_HAAR_MODEL,
        resize_height: int = 300,
    ) -> None:
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This OpenCV build does not provide CascadeClassifier. "
                "Haar is optional and is not part of the OpenCV 5 default path."
            )
        cascade_path = Path(cascade_path).expanduser().resolve()
        if not cascade_path.is_file():
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
        if resize_height <= 0:
            raise ValueError("Haar resize height must be positive.")
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"OpenCV could not load Haar cascade: {cascade_path}")
        self._resize_height = resize_height

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect at a fixed height and scale boxes back to source coordinates."""

        if frame is None or frame.size == 0:
            raise ValueError("Cannot run Haar on an empty frame.")
        height, width = frame.shape[:2]
        resized_width = max(1, round(width * self._resize_height / height))
        small = cv2.resize(frame, (resized_width, self._resize_height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(gray)
        scale_x = width / float(resized_width)
        scale_y = height / float(self._resize_height)

        rows = np.zeros((len(boxes), 15), dtype=np.float32)
        for index, (x, y, box_width, box_height) in enumerate(boxes):
            rows[index, :4] = (
                x * scale_x,
                y * scale_y,
                box_width * scale_x,
                box_height * scale_y,
            )
            rows[index, 14] = 1.0
        return rows


class DlibHogDetector:
    """Optional dlib frontal-face HOG baseline with a lazy dependency import."""

    name = "dlib HOG"

    def __init__(self, resize_height: int = 300) -> None:
        try:
            import dlib
        except ImportError as error:
            raise RuntimeError(
                "dlib is not installed. Install requirements-dlib.txt only "
                "when you want the optional HOG baseline."
            ) from error
        if resize_height <= 0:
            raise ValueError("dlib HOG resize height must be positive.")
        self._detector = dlib.get_frontal_face_detector()
        self._resize_height = resize_height

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run dlib on RGB pixels and convert inclusive corners to x/y/w/h."""

        if frame is None or frame.size == 0:
            raise ValueError("Cannot run dlib HOG on an empty frame.")
        height, width = frame.shape[:2]
        resized_width = max(1, round(width * self._resize_height / height))
        small = cv2.resize(frame, (resized_width, self._resize_height))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        boxes = self._detector(rgb, 0)
        scale_x = width / float(resized_width)
        scale_y = height / float(self._resize_height)

        rows = np.zeros((len(boxes), 15), dtype=np.float32)
        for index, box in enumerate(boxes):
            x1 = box.left() * scale_x
            y1 = box.top() * scale_y
            x2 = (box.right() + 1) * scale_x
            y2 = (box.bottom() + 1) * scale_y
            rows[index, :4] = (x1, y1, x2 - x1, y2 - y1)
            rows[index, 14] = 1.0
        return rows


def create_detector(
    detector_name: str,
    *,
    model_path: Path = DEFAULT_YUNET_MODEL,
    score_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    device: str = "cpu",
) -> Detector:
    """Instantiate a requested detector without importing optional dlib by default."""

    normalized = detector_name.strip().lower()
    if normalized == "yunet":
        return YuNetDetector(
            model_path,
            score_threshold,
            nms_threshold,
            top_k,
            device,
        )
    if normalized == "haar":
        return HaarDetector()
    if normalized in {"hog", "dlib", "dlib-hog"}:
        return DlibHogDetector()
    raise ValueError(
        f"Unknown detector '{detector_name}'. Choose yunet, haar, or hog."
    )


def clipped_box(
    row: np.ndarray,
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Clip one x/y/w/h result to an inclusive drawable image rectangle."""

    height, width = frame_shape[:2]
    x1 = int(np.floor(row[0]))
    y1 = int(np.floor(row[1]))
    x2 = int(np.ceil(row[0] + row[2]))
    y2 = int(np.ceil(row[1] + row[3]))
    x1 = int(np.clip(x1, 0, max(0, width - 1)))
    y1 = int(np.clip(y1, 0, max(0, height - 1)))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def validate_detections(frame: np.ndarray, detections: np.ndarray) -> None:
    """Validate detector shape, finite values, scores, and drawable bounds."""

    if frame is None or frame.size == 0:
        raise RuntimeError("Validation received an empty frame.")
    if detections.ndim != 2 or detections.shape[1] != 15:
        raise RuntimeError(
            f"Expected face rows with 15 values, got {detections.shape}."
        )
    if not np.isfinite(detections).all():
        raise RuntimeError("Face detections contain NaN or infinite values.")
    for row in detections:
        if row[2] <= 0.0 or row[3] <= 0.0:
            raise RuntimeError(f"Face box has non-positive size: {row[:4]}")
        if not 0.0 <= row[14] <= 1.0:
            raise RuntimeError(f"Face score is outside [0, 1]: {row[14]}")
        x1, y1, x2, y2 = clipped_box(row, frame.shape)
        if not (0 <= x1 < x2 <= frame.shape[1]):
            raise RuntimeError("Face box has invalid horizontal bounds.")
        if not (0 <= y1 < y2 <= frame.shape[0]):
            raise RuntimeError("Face box has invalid vertical bounds.")


def draw_detections(
    frame: np.ndarray,
    detections: np.ndarray,
    label: str,
) -> np.ndarray:
    """Draw boxes, YuNet landmarks when present, scores, and a panel label."""

    output = frame.copy()
    for row in detections:
        x1, y1, x2, y2 = clipped_box(row, frame.shape)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            f"{row[14]:.3f}",
            (x1, max(14, y1 + 14)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        # YuNet supplies five finite landmark pairs. Historical detectors leave
        # these ten columns as zeros and therefore skip this drawing branch.
        landmarks = row[4:14].reshape(5, 2)
        if np.any(landmarks):
            for landmark, color in zip(landmarks, LANDMARK_COLORS):
                x = int(round(float(landmark[0])))
                y = int(round(float(landmark[1])))
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(output, (x, y), 2, color, 2, cv2.LINE_AA)

    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"{label}: {len(detections)} face(s)",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def write_image(path: Path, image: np.ndarray) -> None:
    """Create parent directories and fail if OpenCV cannot encode the image."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write output image: {path}")

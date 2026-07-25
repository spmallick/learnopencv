"""Shared OpenCV DNN helpers for the image and video colorization demos."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "colorization_eccv16.onnx"
NETWORK_SIZE = (256, 256)


def load_network(model_path: str | Path = DEFAULT_MODEL) -> cv.dnn.Net:
    """Load the OpenCV 5-compatible ONNX export on the CPU backend."""
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(
            f"Model not found: {model}. Run ./getModels.sh before the demo."
        )

    network = cv.dnn.readNetFromONNX(str(model))
    network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return network


def colorize_frame(frame_bgr: np.ndarray, network: cv.dnn.Net) -> tuple[np.ndarray, float]:
    """Colorize one BGR frame and return it with a simple chroma score."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("The input frame is empty.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected an 8-bit, three-channel BGR image.")

    image = frame_bgr.astype(np.float32) / 255.0
    image_lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    lightness = image_lab[:, :, 0]

    resized_lightness = cv.resize(lightness, NETWORK_SIZE, interpolation=cv.INTER_CUBIC)
    network.setInput(cv.dnn.blobFromImage(resized_lightness))
    predicted_ab = network.forward()
    if predicted_ab.ndim != 4 or predicted_ab.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected network output shape: {tuple(predicted_ab.shape)}"
        )

    ab = predicted_ab[0].transpose(1, 2, 0)
    ab = cv.resize(
        ab,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        interpolation=cv.INTER_CUBIC,
    )
    output_lab = np.concatenate((lightness[:, :, None], ab), axis=2)
    output_bgr = cv.cvtColor(output_lab, cv.COLOR_Lab2BGR)
    output_bgr = np.clip(output_bgr * 255.0, 0, 255).astype(np.uint8)

    chroma_score = float(np.mean(np.linalg.norm(ab, axis=2)))
    return output_bgr, chroma_score


def validate_output(
    input_bgr: np.ndarray,
    output_bgr: np.ndarray,
    chroma_score: float,
) -> None:
    """Raise an informative error when an inference result is not usable."""
    if output_bgr.shape != input_bgr.shape:
        raise RuntimeError(
            f"Output shape {output_bgr.shape} does not match input {input_bgr.shape}."
        )
    if output_bgr.dtype != np.uint8:
        raise RuntimeError(f"Expected uint8 output, got {output_bgr.dtype}.")
    if not np.isfinite(chroma_score) or chroma_score <= 1.0:
        raise RuntimeError(
            f"Colorization produced too little chroma ({chroma_score:.3f})."
        )

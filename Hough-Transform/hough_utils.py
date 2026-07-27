"""Shared Hough line/circle detection functions for Python examples."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent
UInt8Image = npt.NDArray[np.uint8]
IntLines = npt.NDArray[np.int32]
FloatCircles = npt.NDArray[np.float32]


def resolve_input(path: str | Path | None, default_name: str) -> Path:
    """Resolve an explicit path or a bundled input relative to this module."""

    return (
        (PROJECT_DIR / default_name)
        if path is None
        else Path(path).expanduser().resolve()
    )


def read_bgr(path: str | Path) -> UInt8Image:
    """Read a color image and fail clearly when the input cannot be decoded."""

    input_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"could not read image: {input_path}")
    return image


def write_image(path: str | Path, image: UInt8Image) -> Path:
    """Create parent directories, write an image, and return its absolute path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.size == 0 or not cv2.imwrite(str(output_path), image):
        raise OSError(f"could not write image: {output_path}")
    return output_path


def normalize_lines(lines: npt.ArrayLike | None) -> IntLines:
    """Return line endpoints as a deterministic, lexicographically sorted Nx4 array."""

    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    normalized = np.asarray(lines, dtype=np.int32).reshape(-1, 4).copy()
    for row in normalized:
        if (int(row[2]), int(row[3])) < (int(row[0]), int(row[1])):
            row[:] = (row[2], row[3], row[0], row[1])
    order = np.lexsort(
        (normalized[:, 3], normalized[:, 2], normalized[:, 1], normalized[:, 0])
    )
    return normalized[order]


def detect_lines(
    image_bgr: UInt8Image,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 50,
    min_line_length: int = 40,
    max_line_gap: int = 25,
) -> tuple[UInt8Image, IntLines]:
    """Detect probabilistic Hough line segments from a BGR image."""

    if not 0 <= canny_low < canny_high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    if hough_threshold <= 0 or min_line_length < 0 or max_line_gap < 0:
        raise ValueError("Hough threshold/length/gap parameters must be nonnegative")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    raw_lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 180.0,
        hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    return edges, normalize_lines(raw_lines)


def draw_lines(image_bgr: UInt8Image, lines: IntLines) -> UInt8Image:
    """Draw sorted line segments in red on a copy of the BGR input."""

    output = image_bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def normalize_circles(circles: npt.ArrayLike | None) -> FloatCircles:
    """Return circles as a deterministic Nx3 float32 array sorted by x/y/r."""

    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    normalized = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    order = np.lexsort((normalized[:, 2], normalized[:, 1], normalized[:, 0]))
    return normalized[order]


def detect_circles(
    image_bgr: UInt8Image,
    *,
    dp: float = 1.2,
    min_distance: float = 20.0,
    param1: float = 120.0,
    param2: float = 30.0,
    min_radius: int = 20,
    max_radius: int = 60,
) -> tuple[UInt8Image, FloatCircles]:
    """Detect circles with OpenCV's gradient Hough method."""

    if dp < 1.0 or min_distance <= 0:
        raise ValueError("dp must be >= 1 and min_distance must be positive")
    if param1 <= 0 or param2 <= 0:
        raise ValueError("param1 and param2 must be positive")
    if min_radius < 0 or max_radius < min_radius:
        raise ValueError("radius bounds must satisfy 0 <= min <= max")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    raw_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp,
        min_distance,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    return blurred, normalize_circles(raw_circles)


def draw_circles(image_bgr: UInt8Image, circles: FloatCircles) -> UInt8Image:
    """Draw detected circle outlines and centers on a BGR copy."""

    output = image_bgr.copy()
    for x_value, y_value, radius_value in circles:
        center = (int(round(float(x_value))), int(round(float(y_value))))
        radius = int(round(float(radius_value)))
        cv2.circle(output, center, radius, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1, cv2.LINE_AA)
    return output


def line_output_paths(output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Return deterministic edge and annotated-line output paths."""

    directory = Path(output_dir).expanduser().resolve()
    return directory / f"{stem}-edges.png", directory / f"{stem}-lines.png"


def circle_output_paths(output_dir: str | Path, stem: str) -> tuple[Path, Path]:
    """Return deterministic blurred-gray and annotated-circle output paths."""

    directory = Path(output_dir).expanduser().resolve()
    return directory / f"{stem}-blurred.png", directory / f"{stem}-circles.png"

"""Shared, testable color-space operations for the tutorial programs."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

PROJECT_DIR: Final[Path] = Path(__file__).resolve().parent
COLOR_CONVERSIONS: Final[dict[str, int | None]] = {
    "BGR": None,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
    "Lab": cv2.COLOR_BGR2Lab,
}

UInt8Image = npt.NDArray[np.uint8]


def default_cube_images() -> list[Path]:
    """Return the bundled cube images in deterministic filename order."""

    return sorted((PROJECT_DIR / "images").glob("rub*.jpg"))


def resolve_input_path(path: str | Path | None) -> Path:
    """Resolve an explicit path or return the first bundled cube image."""

    if path is None:
        images = default_cube_images()
        if not images:
            raise FileNotFoundError("no bundled images/rub*.jpg files were found")
        return images[0]
    return Path(path).expanduser().resolve()


def read_bgr(path: str | Path) -> UInt8Image:
    """Read a color image and raise a clear error when decoding fails."""

    image_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return image


def convert_bgr(image: UInt8Image, color_space: str) -> UInt8Image:
    """Convert a BGR image to one of the tutorial's supported encodings."""

    if color_space not in COLOR_CONVERSIONS:
        choices = ", ".join(COLOR_CONVERSIONS)
        raise ValueError(f"unsupported color space {color_space!r}; choose {choices}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an 8-bit, three-channel BGR array")

    conversion = COLOR_CONVERSIONS[color_space]
    return image.copy() if conversion is None else cv2.cvtColor(image, conversion)


def convert_pixel(bgr: npt.ArrayLike) -> dict[str, tuple[int, int, int]]:
    """Return one 8-bit BGR pixel represented in all supported color spaces."""

    pixel = np.asarray(bgr, dtype=np.uint8)
    if pixel.shape != (3,):
        raise ValueError(f"BGR pixel must have shape (3,), got {pixel.shape}")
    pixel_image = pixel.reshape(1, 1, 3)

    values: dict[str, tuple[int, int, int]] = {}
    for name in COLOR_CONVERSIONS:
        converted = convert_bgr(pixel_image, name)[0, 0]
        values[name] = tuple(int(value) for value in converted)
    return values


def _threshold_vector(values: npt.ArrayLike, label: str) -> npt.NDArray[np.int32]:
    """Validate and normalize a three-channel threshold vector."""

    vector = np.asarray(values, dtype=np.int32)
    if vector.shape != (3,):
        raise ValueError(f"{label} threshold must contain exactly three values")
    if np.any(vector < 0) or np.any(vector > 255):
        raise ValueError(f"{label} threshold values must be in [0, 255]")
    return vector


def threshold_mask(
    image_bgr: UInt8Image,
    color_space: str,
    lower: npt.ArrayLike,
    upper: npt.ArrayLike,
) -> UInt8Image:
    """Create an inclusive three-channel mask in the requested color space.

    For 8-bit HSV, hue uses OpenCV's 0-179 encoding. A lower hue greater than
    the upper hue requests a wrapped range, such as 170-179 plus 0-10 for red.
    """

    converted = convert_bgr(image_bgr, color_space)
    lower_vector = _threshold_vector(lower, "lower")
    upper_vector = _threshold_vector(upper, "upper")

    if color_space == "HSV":
        if lower_vector[0] > 179 or upper_vector[0] > 179:
            raise ValueError("8-bit HSV hue thresholds must be in [0, 179]")
        if np.any(lower_vector[1:] > upper_vector[1:]):
            raise ValueError("HSV saturation/value lower bounds must not exceed upper bounds")
        if lower_vector[0] > upper_vector[0]:
            first_lower = lower_vector.copy()
            first_upper = upper_vector.copy()
            first_upper[0] = 179
            second_lower = lower_vector.copy()
            second_lower[0] = 0
            first = cv2.inRange(converted, first_lower, first_upper)
            second = cv2.inRange(converted, second_lower, upper_vector)
            return cv2.bitwise_or(first, second)
    elif np.any(lower_vector > upper_vector):
        raise ValueError("lower threshold values must not exceed upper values")

    return cv2.inRange(converted, lower_vector, upper_vector)


def apply_mask(image_bgr: UInt8Image, mask: UInt8Image) -> UInt8Image:
    """Apply a one-channel mask while keeping the displayed image in BGR."""

    if mask.shape != image_bgr.shape[:2] or mask.dtype != np.uint8:
        raise ValueError("mask must be uint8 and match the image height and width")
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def write_image(path: str | Path, image: UInt8Image) -> Path:
    """Create parent directories, encode an image, and return its absolute path."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.size == 0 or not cv2.imwrite(str(output_path), image):
        raise OSError(f"could not write image: {output_path}")
    return output_path


def run_core_validation() -> dict[str, int]:
    """Validate bundled inputs and stable one-pixel conversion values."""

    cube_images = default_cube_images()
    piece_images = sorted((PROJECT_DIR / "pieces").glob("*.jpg"))
    if len(cube_images) != 10:
        raise AssertionError(f"expected 10 cube images, found {len(cube_images)}")
    if len(piece_images) != 56:
        raise AssertionError(f"expected 56 cropped piece images, found {len(piece_images)}")
    for path in (*cube_images, *piece_images):
        read_bgr(path)

    expected = {
        "BGR": (40, 158, 16),
        "HSV": (65, 229, 158),
        "YCrCb": (102, 67, 93),
        "Lab": (145, 71, 177),
    }
    actual = convert_pixel((40, 158, 16))
    if actual != expected:
        raise AssertionError(f"unexpected pixel conversions: {actual}")
    return {"cube_images": len(cube_images), "piece_images": len(piece_images)}

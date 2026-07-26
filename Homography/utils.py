"""Shared, testable utilities for the homography examples."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


def read_image(path: Path | str) -> np.ndarray:
    """Read a color image and raise a clear error when it cannot be decoded."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: Path | str, image: np.ndarray) -> None:
    """Write an image, creating its parent directory when needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Could not write image: {output}")


def parse_points(value: str) -> np.ndarray:
    """Parse ``x,y;x,y;x,y;x,y`` into a four-point float32 array."""
    try:
        rows = [
            [float(coordinate.strip()) for coordinate in pair.split(",")]
            for pair in value.split(";")
        ]
    except ValueError as error:
        raise ValueError(
            "Points must use the format x,y;x,y;x,y;x,y."
        ) from error
    if len(rows) != 4 or any(len(row) != 2 for row in rows):
        raise ValueError("Exactly four x,y point pairs are required.")
    return validate_quad(np.asarray(rows, dtype=np.float32))


def validate_quad(
    points: np.ndarray,
    image_shape: tuple[int, ...] | None = None,
    *,
    name: str = "points",
) -> np.ndarray:
    """Validate an ordered, convex quadrilateral and optional image bounds."""
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError(f"{name} must have shape (4, 2), not {quad.shape}.")
    if not np.isfinite(quad).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    if abs(cv2.contourArea(quad)) < 1.0:
        raise ValueError(f"{name} must enclose a non-zero area.")
    if not cv2.isContourConvex(quad):
        raise ValueError(
            f"{name} must be ordered around a convex quadrilateral."
        )

    if image_shape is not None:
        height, width = image_shape[:2]
        x_coordinates, y_coordinates = quad[:, 0], quad[:, 1]
        if (
            np.any(x_coordinates < 0)
            or np.any(x_coordinates >= width)
            or np.any(y_coordinates < 0)
            or np.any(y_coordinates >= height)
        ):
            raise ValueError(
                f"{name} must lie within the {width}x{height} image."
            )
    return quad


def image_corners(image: np.ndarray) -> np.ndarray:
    """Return image corners in top-left clockwise order."""
    if image is None or image.size == 0:
        raise ValueError("The image must not be empty.")
    height, width = image.shape[:2]
    return rectangle_corners(width, height)


def rectangle_corners(width: int, height: int) -> np.ndarray:
    """Return corners for a rectangle of the requested pixel dimensions."""
    if width < 2 or height < 2:
        raise ValueError("Width and height must both be at least 2 pixels.")
    return np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )


def compute_homography(
    source_points: np.ndarray, destination_points: np.ndarray
) -> np.ndarray:
    """Compute a homography from two validated four-point quads."""
    source = validate_quad(source_points, name="source_points")
    destination = validate_quad(
        destination_points, name="destination_points"
    )
    homography, inlier_mask = cv2.findHomography(source, destination, method=0)
    if homography is None or inlier_mask is None:
        raise ValueError("OpenCV could not compute a homography.")
    return homography


def rectify_image(
    image: np.ndarray,
    source_points: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a source quadrilateral into a width-by-height image."""
    source = validate_quad(
        source_points, image.shape, name="source_points"
    )
    destination = rectangle_corners(width, height)
    homography = compute_homography(source, destination)
    rectified = cv2.warpPerspective(image, homography, (width, height))
    return rectified, homography


def composite_on_quad(
    source: np.ndarray,
    destination: np.ndarray,
    destination_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ``source`` into a destination quad and return image and matrix."""
    destination_quad = validate_quad(
        destination_points,
        destination.shape,
        name="destination_points",
    )
    homography = compute_homography(
        image_corners(source), destination_quad
    )
    output_size = (destination.shape[1], destination.shape[0])
    warped_source = cv2.warpPerspective(source, homography, output_size)

    source_mask = np.full(source.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        source_mask,
        homography,
        output_size,
        flags=cv2.INTER_NEAREST,
    )
    result = destination.copy()
    cv2.copyTo(warped_source, warped_mask, result)
    return result, homography


def mouse_handler(
    event: int, x: int, y: int, flags: int, data: dict
) -> None:
    """Collect at most four left-clicks for ``get_four_points``."""
    del flags
    if event == cv2.EVENT_LBUTTONDOWN and len(data["points"]) < 4:
        data["points"].append([x, y])
        cv2.circle(
            data["preview"],
            (x, y),
            4,
            (0, 0, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )


def get_four_points(
    image: np.ndarray, window_name: str = "Select four points"
) -> np.ndarray:
    """Interactively collect four points, with Enter to accept and R to reset."""
    if image is None or image.size == 0:
        raise ValueError("The image must not be empty.")

    data = {"preview": image.copy(), "points": []}
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_handler, data)
        while True:
            preview = data["preview"].copy()
            cv2.putText(
                preview,
                f"Points: {len(data['points'])}/4 | Enter: accept | R: reset | Esc: cancel",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13, 32) and len(data["points"]) == 4:
                break
            if key in (ord("r"), ord("R")):
                data = {"preview": image.copy(), "points": []}
                cv2.setMouseCallback(window_name, mouse_handler, data)
            if key == 27:
                raise RuntimeError("Point selection was cancelled.")
    except cv2.error as error:
        raise RuntimeError(
            "Interactive point selection requires an OpenCV GUI. "
            "Use --points for a headless run."
        ) from error
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    return validate_quad(
        np.asarray(data["points"], dtype=np.float32),
        image.shape,
        name="selected points",
    )


def show_images(images: Mapping[str, np.ndarray]) -> None:
    """Display named images until a key is pressed."""
    try:
        for name, image in images.items():
            cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as error:
        raise RuntimeError(
            "Display requires an OpenCV GUI; omit --display in headless runs."
        ) from error

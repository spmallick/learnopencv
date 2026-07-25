#!/usr/bin/env python3
"""Estimate a six-point head pose with OpenCV solvePnP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

IMAGE_POINTS = np.array(
    [
        (359.0, 391.0),  # Nose tip
        (399.0, 561.0),  # Chin
        (337.0, 297.0),  # Left eye, left corner
        (513.0, 301.0),  # Right eye, right corner
        (345.0, 465.0),  # Left mouth corner
        (453.0, 469.0),  # Right mouth corner
    ],
    dtype=np.float64,
)

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def camera_matrix_for(image: np.ndarray, focal_length: float | None = None) -> np.ndarray:
    """Build the simple pinhole camera matrix used by the tutorial."""
    height, width = image.shape[:2]
    focal = float(width if focal_length is None else focal_length)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_vector_to_euler_degrees(rotation_vector: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to XYZ Euler angles in degrees."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.hypot(rotation_matrix[0, 0], rotation_matrix[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def estimate_pose(
    image: np.ndarray, focal_length: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the camera matrix, rotation, translation, and reprojection RMSE."""
    camera_matrix = camera_matrix_for(image, focal_length)
    distortion = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        IMAGE_POINTS,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP could not estimate a pose")

    reprojected, _ = cv2.projectPoints(
        MODEL_POINTS, rotation_vector, translation_vector, camera_matrix, distortion
    )
    residuals = reprojected.reshape(-1, 2) - IMAGE_POINTS
    rmse = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
    return camera_matrix, rotation_vector, translation_vector, rmse


def draw_pose(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_vector: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    """Draw the six landmarks and a line pointing out from the nose."""
    result = image.copy()
    distortion = np.zeros((4, 1), dtype=np.float64)
    nose_end, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)], dtype=np.float64),
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )

    for x, y in IMAGE_POINTS:
        cv2.circle(result, (round(x), round(y)), 4, (0, 0, 255), -1, cv2.LINE_AA)
    start = tuple(np.rint(IMAGE_POINTS[0]).astype(int))
    end = tuple(np.rint(nose_end.reshape(2)).astype(int))
    cv2.line(result, start, end, (255, 0, 0), 3, cv2.LINE_AA)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "headPose.jpg")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--focal-length", type=float)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Fail if the result is invalid or reprojection RMSE exceeds 50 pixels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    camera, rotation, translation, rmse = estimate_pose(image, args.focal_length)
    euler = rotation_vector_to_euler_degrees(rotation)
    result = draw_pose(image, camera, rotation, translation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "head-pose-result.jpg"
    metrics_path = args.output_dir / "head-pose-metrics.json"
    if not cv2.imwrite(str(image_path), result):
        raise RuntimeError(f"Could not write result: {image_path}")

    metrics = {
        "opencv_version": cv2.__version__,
        "rotation_vector": rotation.reshape(-1).tolist(),
        "translation_vector": translation.reshape(-1).tolist(),
        "euler_xyz_degrees": euler.tolist(),
        "reprojection_rmse_pixels": rmse,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"OpenCV: {cv2.__version__}")
    print(f"Camera matrix:\n{camera}")
    print(f"Rotation vector:\n{rotation}")
    print(f"Translation vector:\n{translation}")
    print(f"Euler XYZ (degrees): {euler}")
    print(f"Reprojection RMSE: {rmse:.3f} pixels")
    print(f"Saved: {image_path}")

    if args.validate:
        values = np.concatenate((rotation.ravel(), translation.ravel(), [rmse]))
        if not np.isfinite(values).all() or rmse > 50.0:
            raise RuntimeError("Head-pose validation failed")
        check = cv2.imread(str(image_path))
        if check is None or check.shape != image.shape:
            raise RuntimeError("Saved output validation failed")
        print("Validation: PASS")

    if not args.no_display:
        cv2.imshow("Head pose", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

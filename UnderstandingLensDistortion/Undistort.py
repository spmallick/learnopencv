"""Calibrate a camera and save corrected images without requiring a GUI."""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATTERN = str(SCRIPT_DIR / "images" / "*.jpg")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


@dataclass(frozen=True)
class CalibrationResult:
    rms: float
    reprojection_rmse: float
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: tuple[np.ndarray, ...]
    translation_vectors: tuple[np.ndarray, ...]
    image_size: tuple[int, int]
    successful_images: tuple[Path, ...]
    failed_images: tuple[Path, ...]
    corner_preview: np.ndarray


@dataclass(frozen=True)
class UndistortionResult:
    image: np.ndarray
    new_camera_matrix: np.ndarray
    roi: tuple[int, int, int, int]


def discover_images(pattern: str = DEFAULT_PATTERN) -> list[Path]:
    paths = sorted(Path(item).resolve() for item in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No calibration images match: {pattern}")
    return paths


def _object_template(
    board_size: tuple[int, int], square_size: float
) -> np.ndarray:
    columns, rows = board_size
    if columns < 2 or rows < 2:
        raise ValueError("Checkerboard dimensions must both be at least 2.")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be finite and positive.")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    )
    return points


def calibrate_from_images(
    image_paths: Sequence[str | Path],
    *,
    board_size: tuple[int, int] = (6, 9),
    square_size: float = 1.0,
    require_all: bool = False,
) -> CalibrationResult:
    if not image_paths:
        raise ValueError("At least one calibration image is required.")

    object_template = _object_template(board_size, square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    successful: list[Path] = []
    failed: list[Path] = []
    image_size: tuple[int, int] | None = None
    corner_preview: np.ndarray | None = None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for item in sorted(Path(path).expanduser().resolve() for path in image_paths):
        image = cv2.imread(str(item), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read calibration image: {item}")
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = (grayscale.shape[1], grayscale.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                f"Calibration image {item} has size {current_size}; "
                f"expected {image_size}."
            )

        found, corners = cv2.findChessboardCorners(
            grayscale, board_size, flags
        )
        if not found:
            failed.append(item)
            continue
        refined = cv2.cornerSubPix(
            grayscale, corners, (11, 11), (-1, -1), criteria
        )
        object_points.append(object_template.copy())
        image_points.append(refined)
        successful.append(item)
        if corner_preview is None:
            corner_preview = image.copy()
            cv2.drawChessboardCorners(
                corner_preview, board_size, refined, True
            )

    if require_all and failed:
        raise RuntimeError(
            f"Checkerboard detection failed for {len(failed)} image(s)."
        )
    if len(successful) < 3 or image_size is None or corner_preview is None:
        raise RuntimeError(
            "At least three successful checkerboard views are required."
        )

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )

    squared_error = 0.0
    point_count = 0
    for object_view, image_view, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            object_view, rvec, tvec, camera_matrix, distortion
        )
        difference = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        squared_error += float(np.sum(difference * difference))
        point_count += difference.shape[0]
    reprojection_rmse = float(np.sqrt(squared_error / point_count))

    return CalibrationResult(
        rms=float(rms),
        reprojection_rmse=reprojection_rmse,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        rotation_vectors=tuple(rvecs),
        translation_vectors=tuple(tvecs),
        image_size=image_size,
        successful_images=tuple(successful),
        failed_images=tuple(failed),
        corner_preview=corner_preview,
    )


def undistort_image(
    image: np.ndarray,
    calibration: CalibrationResult,
    *,
    alpha: float = 1.0,
    method: str = "direct",
    crop: bool = False,
) -> UndistortionResult:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.shape[1::-1] != calibration.image_size:
        raise ValueError(
            f"Input image size {image.shape[1::-1]} does not match "
            f"calibration size {calibration.image_size}."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        calibration.image_size,
        alpha,
        calibration.image_size,
    )
    if method == "direct":
        corrected = cv2.undistort(
            image,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            new_camera_matrix,
        )
    elif method == "remap":
        map_x, map_y = cv2.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            new_camera_matrix,
            calibration.image_size,
            cv2.CV_32FC1,
        )
        corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    else:
        raise ValueError("method must be 'direct' or 'remap'.")

    normalized_roi = tuple(int(value) for value in roi)
    if crop:
        x, y, width, height = normalized_roi
        if width <= 0 or height <= 0:
            raise RuntimeError("OpenCV returned an empty undistortion ROI.")
        corrected = corrected[y : y + height, x : x + width].copy()
    return UndistortionResult(
        corrected, new_camera_matrix, normalized_roi
    )


def save_calibration(
    calibration: CalibrationResult, path: str | Path
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(destination), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Could not write calibration: {destination}")
    try:
        storage.write("imageWidth", calibration.image_size[0])
        storage.write("imageHeight", calibration.image_size[1])
        storage.write("rms", calibration.rms)
        storage.write("reprojectionRMSE", calibration.reprojection_rmse)
        storage.write("cameraMatrix", calibration.camera_matrix)
        storage.write(
            "distortionCoefficients",
            calibration.distortion_coefficients,
        )
    finally:
        storage.release()
    return destination


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write output image: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=DEFAULT_PATTERN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--board-columns", type=int, default=6)
    parser.add_argument("--board-rows", type=int, default=9)
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the saved result after calibration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = discover_images(args.images)
    calibration = calibrate_from_images(
        paths,
        board_size=(args.board_columns, args.board_rows),
        square_size=args.square_size,
        require_all=args.require_all,
    )
    sample = cv2.imread(
        str(calibration.successful_images[0]), cv2.IMREAD_COLOR
    )
    if sample is None:
        raise FileNotFoundError(
            f"Could not reread sample: {calibration.successful_images[0]}"
        )
    direct = undistort_image(
        sample, calibration, alpha=args.alpha, method="direct", crop=args.crop
    )
    remapped = undistort_image(
        sample, calibration, alpha=args.alpha, method="remap", crop=args.crop
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_image(output_dir / "calibration-corners.jpg", calibration.corner_preview)
    _write_image(output_dir / "undistorted-direct.jpg", direct.image)
    _write_image(output_dir / "undistorted-remap.jpg", remapped.image)
    calibration_path = save_calibration(
        calibration, output_dir / "calibration.yml"
    )

    print(
        f"Checkerboards detected: {len(calibration.successful_images)}/"
        f"{len(paths)}"
    )
    print(f"Image size: {calibration.image_size[0]}x{calibration.image_size[1]}")
    print(f"OpenCV calibration RMS: {calibration.rms:.10f}")
    print(f"Reprojection RMSE: {calibration.reprojection_rmse:.10f} px")
    print(f"Alpha-{args.alpha:g} ROI: {direct.roi}")
    print(f"Saved calibration: {calibration_path}")
    print(f"Saved corrected images under: {output_dir}")

    if args.show:
        cv2.imshow("Calibration corners", calibration.corner_preview)
        cv2.imshow("Undistorted image", direct.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

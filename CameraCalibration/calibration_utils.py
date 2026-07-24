"""Reusable camera-calibration helpers for OpenCV 4.14 and OpenCV 5.0."""

from __future__ import annotations

# Python's glob module accepts the same wildcard-style input exposed by the CLI.
import glob
# A dataclass gives the tutorial a named result instead of an error-prone tuple.
from dataclasses import dataclass
# Path keeps input and output handling independent of the caller's directory.
from pathlib import Path

# OpenCV supplies checkerboard detection, calibration, and undistortion.
import cv2
# NumPy constructs object points and validates numeric results.
import numpy as np


# The printed checkerboard has six inner corners across and nine down.
CHECKERBOARD = (6, 9)
# Sub-pixel refinement stops after 30 iterations or 0.001-pixel convergence.
CORNER_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
# These flags normalize lighting, adapt the threshold, and reject obvious
# non-checkerboards quickly. They are available in both supported OpenCV majors.
CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_FAST_CHECK
    | cv2.CALIB_CB_NORMALIZE_IMAGE
)


@dataclass(frozen=True)
class CalibrationResult:
    """All values needed to inspect or reuse one calibration."""

    # OpenCV's optimizer reports its own root-mean-square pixel error.
    rms_error: float
    # The tutorial independently recomputes the reprojection RMSE.
    reprojection_rmse: float
    # This 3-by-3 matrix stores focal lengths and the optical center.
    camera_matrix: np.ndarray
    # These coefficients model radial and tangential lens distortion.
    distortion_coefficients: np.ndarray
    # One rotation vector maps the checkerboard pose for each valid image.
    rotation_vectors: tuple[np.ndarray, ...]
    # One translation vector accompanies each rotation vector.
    translation_vectors: tuple[np.ndarray, ...]
    # OpenCV represents image size as (width, height), not array shape order.
    image_size: tuple[int, int]
    # Keeping the complete match list makes input coverage testable.
    image_paths: tuple[Path, ...]
    # Some readable images might not contain a detectable checkerboard.
    detected_images: int


def calibrate_images(image_pattern: str, display: bool = True) -> CalibrationResult:
    """Detect checkerboards and calibrate a camera from matching images."""

    # Sorting makes the default sample and printed results reproducible.
    image_paths = tuple(Path(path) for path in sorted(glob.glob(image_pattern)))
    # Calibration cannot proceed without at least one readable candidate.
    if not image_paths:
        raise FileNotFoundError(f"No calibration images matched: {image_pattern}")

    # Each checkerboard square is one arbitrary world unit apart on the z=0 plane.
    object_template = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3), dtype=np.float32
    )
    # mgrid generates column/row coordinates in the same order as corner detection.
    object_template[:, :2] = np.mgrid[
        0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]
    ].T.reshape(-1, 2)

    # Calibration needs one copy of the known 3D grid per successful image.
    object_points: list[np.ndarray] = []
    # The corresponding refined 2D pixel coordinates come from each image.
    image_points: list[np.ndarray] = []
    # The first image establishes the size required for every later image.
    image_size: tuple[int, int] | None = None

    try:
        # Process every matched file so missing or inconsistent assets fail clearly.
        for image_path in image_paths:
            # Preserve the color image for visualization and the undistortion lesson.
            image = cv2.imread(str(image_path))
            # imread returns None instead of raising when decoding fails.
            if image is None:
                raise OSError(f"Could not read calibration image: {image_path}")

            # Corner detection operates on one intensity channel.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Convert NumPy's (height, width) shape to OpenCV's (width, height).
            current_size = (gray.shape[1], gray.shape[0])
            # Record the first size, then reject mixed-resolution calibration sets.
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise ValueError(
                    "All calibration images must have the same size; "
                    f"{image_path} is {current_size}, expected {image_size}"
                )

            # Locate the requested six-by-nine grid of inner checkerboard corners.
            found, corners = cv2.findChessboardCorners(
                gray, CHECKERBOARD, CHESSBOARD_FLAGS
            )
            # Only successful detections contribute point correspondences.
            if found:
                # Refine integer-scale detections to sub-pixel coordinates.
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA
                )
                # Copy the template so every view owns an independent array.
                object_points.append(object_template.copy())
                # Keep OpenCV's corner array for calibration.
                image_points.append(refined_corners)
                # Overlay the refined corners for the interactive teaching view.
                cv2.drawChessboardCorners(
                    image, CHECKERBOARD, refined_corners, found
                )

            # Headless validation skips all window-system calls.
            if display:
                cv2.imshow("Checkerboard corners", image)
                # Pause so the reader can inspect each image before advancing.
                cv2.waitKey(0)
    finally:
        # Also close an already-open window if an input fails partway through.
        if display:
            cv2.destroyAllWindows()

    # Three distinct views are a practical minimum for this tutorial.
    if len(object_points) < 3:
        raise RuntimeError(
            "Calibration needs at least 3 valid checkerboards; "
            f"found {len(object_points)}"
        )
    # A nonempty match list always initializes image_size before this point.
    assert image_size is not None

    # Estimate intrinsics, distortion, and one board pose per detected image.
    (
        rms_error,
        camera_matrix,
        distortion,
        rotation_vectors,
        translation_vectors,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    # Accumulate squared pixel residuals for an independent RMSE calculation.
    squared_error = 0.0
    # Divide by the number of observed 2D points rather than by image count.
    point_count = 0
    # Reproject every known 3D grid using the optimized pose and camera model.
    for object_set, observed, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(
            object_set,
            rotation,
            translation,
            camera_matrix,
            distortion,
        )
        # OpenCV 4.14 returns corners as (N, 1, 2), while OpenCV 5.0 returns
        # (N, 2). Flattening the singleton dimension preserves identical points.
        observed_xy = observed.reshape(-1, 2).astype(np.float64)
        # projectPoints can likewise vary only in representation, not semantics.
        projected_xy = projected.reshape(-1, 2).astype(np.float64)
        # Convert both arrays before subtraction to avoid float32 accumulation.
        point_delta = observed_xy - projected_xy
        # Sum both x and y residuals for every detected corner.
        squared_error += float(np.square(point_delta).sum())
        # Each object point has exactly one projected and observed image point.
        point_count += len(object_set)

    # The square root returns the error in pixels.
    reprojection_rmse = float(np.sqrt(squared_error / point_count))
    # Package both OpenCV outputs and independently testable metadata.
    return CalibrationResult(
        rms_error=float(rms_error),
        reprojection_rmse=reprojection_rmse,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        rotation_vectors=tuple(rotation_vectors),
        translation_vectors=tuple(translation_vectors),
        image_size=image_size,
        image_paths=image_paths,
        detected_images=len(object_points),
    )


def print_calibration(result: CalibrationResult) -> None:
    """Print the calibration values that readers typically inspect first."""

    # Start with coverage so a plausible matrix cannot hide missing images.
    print(
        f"images={len(result.image_paths)}, detections={result.detected_images}, "
        f"image_size={result.image_size}"
    )
    # Print both optimizer and independently derived errors for comparison.
    print(f"RMS calibration error: {result.rms_error:.9f} pixels")
    print(f"Reprojection RMSE: {result.reprojection_rmse:.9f} pixels")
    # The matrix and distortion terms define the reusable intrinsic model.
    print("Camera matrix:\n", result.camera_matrix)
    print("Distortion coefficients:\n", result.distortion_coefficients)
    # Preserve the original tutorial's per-image pose output. Each index refers
    # to the same successful checkerboard view in both collections.
    print("Rotation vectors:")
    for index, rotation_vector in enumerate(result.rotation_vectors):
        print(f"[{index}]\n{rotation_vector}")
    print("Translation vectors:")
    for index, translation_vector in enumerate(result.translation_vectors):
        print(f"[{index}]\n{translation_vector}")


def validate_calibration(result: CalibrationResult) -> None:
    """Validate version-independent numerical invariants for CI runs."""

    # Guard the minimum again so callers can validate a stored result.
    if result.detected_images < 3:
        raise RuntimeError("At least three checkerboards must be detected")
    # Every detected board must have one rotation and one translation estimate.
    if (
        len(result.rotation_vectors) != result.detected_images
        or len(result.translation_vectors) != result.detected_images
    ):
        raise RuntimeError("Calibration pose counts do not match the detections")
    # Intrinsics are always a finite, normalized 3-by-3 matrix.
    if (
        result.camera_matrix.shape != (3, 3)
        or not np.isfinite(result.camera_matrix).all()
        or not np.isclose(result.camera_matrix[2, 2], 1.0)
    ):
        raise RuntimeError("Camera matrix has an invalid shape, range, or scale")
    # Physically meaningful focal lengths are strictly positive.
    if result.camera_matrix[0, 0] <= 0 or result.camera_matrix[1, 1] <= 0:
        raise RuntimeError("Focal lengths must be positive")
    # Distortion parameters must remain finite before they are used for remapping.
    if not np.isfinite(result.distortion_coefficients).all():
        raise RuntimeError("Distortion coefficients contain non-finite values")
    # The optimizer's error must be finite and comfortably below one pixel here.
    if not np.isfinite(result.rms_error) or result.rms_error >= 1.0:
        raise RuntimeError(
            f"Expected calibration RMS below 1 pixel, got {result.rms_error}"
        )
    # The independently accumulated error should meet the same quality bound.
    if not np.isfinite(result.reprojection_rmse) or result.reprojection_rmse >= 1.0:
        raise RuntimeError(
            "Expected reprojection RMSE below 1 pixel, "
            f"got {result.reprojection_rmse}"
        )
    # Both errors describe the same residuals and should agree within rounding.
    if abs(result.rms_error - result.reprojection_rmse) >= 1e-5:
        raise RuntimeError(
            "OpenCV RMS and independently calculated reprojection RMSE disagree"
        )


def undistort_sample(
    result: CalibrationResult,
    sample_path: Path,
    output_dir: Path,
    display: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Undistort one image through direct and precomputed-map workflows."""

    # Read the original pixels rather than a checkerboard-overlay visualization.
    image = cv2.imread(str(sample_path))
    # imread signals a bad path or decode failure with None.
    if image is None:
        raise OSError(f"Could not read image to undistort: {sample_path}")
    # NumPy shape stores height before width.
    height, width = image.shape[:2]
    # Reusing calibration at another resolution requires explicit rescaling.
    if (width, height) != result.image_size:
        raise ValueError(
            f"Undistortion image is {(width, height)}, expected {result.image_size}"
        )

    # Alpha=1 retains all source pixels, including invalid border regions.
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        result.camera_matrix,
        result.distortion_coefficients,
        result.image_size,
        1,
        result.image_size,
    )

    # The first documented method computes the mapping and samples in one call.
    direct = cv2.undistort(
        image,
        result.camera_matrix,
        result.distortion_coefficients,
        None,
        new_camera_matrix,
    )
    # The second method precomputes floating-point maps for efficient reuse.
    map_x, map_y = cv2.initUndistortRectifyMap(
        result.camera_matrix,
        result.distortion_coefficients,
        None,
        new_camera_matrix,
        result.image_size,
        cv2.CV_32FC1,
    )
    # Linear interpolation matches the sampling used by the direct workflow.
    remapped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    # Create the requested destination before attempting either write.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Fixed names let tests and readers distinguish the two methods.
    rendered_outputs = (
        ("direct", direct),
        ("remap", remapped),
    )
    # Check imwrite's boolean result because it does not always raise on failure.
    for method_name, rendered in rendered_outputs:
        output_path = output_dir / f"undistorted_{method_name}.jpg"
        if not cv2.imwrite(str(output_path), rendered):
            raise OSError(f"Could not write output image: {output_path}")

    # Interactive mode shows both results side by side for visual comparison.
    if display:
        cv2.imshow("Undistorted image (direct)", direct)
        cv2.imshow("Undistorted image (remap)", remapped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Version-specific interpolation details can change a small number of pixels.
    mean_difference = float(cv2.absdiff(direct, remapped).mean())
    # Return images as well as the scalar so tests can inspect their full shape.
    return direct, remapped, mean_difference

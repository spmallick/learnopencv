"""Stabilize a video by smoothing feature-based inter-frame camera motion."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "video.mp4"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_SMOOTHING_RADIUS = 50


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    """Return an edge-padded moving average with the same length as *curve*."""
    if radius < 0:
        raise ValueError("The smoothing radius must be non-negative.")
    if curve.size == 0 or radius == 0:
        return curve.copy()

    # Edge padding keeps the filter defined at the beginning and end of the clip.
    padded = np.pad(curve, (radius, radius), mode="edge")
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def smooth_trajectory(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Smooth the x translation, y translation, and rotation independently."""
    smoothed = trajectory.copy()
    for component in range(trajectory.shape[1]):
        smoothed[:, component] = moving_average(
            trajectory[:, component], radius
        )
    return smoothed


def fix_border(frame: np.ndarray) -> np.ndarray:
    """Scale a stabilized frame around its center to hide moving black borders."""
    height, width = frame.shape[:2]
    transform = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), 0.0, 1.04
    )
    return cv2.warpAffine(frame, transform, (width, height))


def identity_transform() -> np.ndarray:
    """Return a 2-by-3 affine identity transform."""
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64)


def estimate_transforms(
    capture: cv2.VideoCapture,
) -> tuple[list[np.ndarray], list[int]]:
    """Estimate partial-affine motion between each pair of decoded frames."""
    ok, previous = capture.read()
    if not ok or previous is None:
        raise RuntimeError("The input video does not contain a readable frame.")

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    transforms: list[np.ndarray] = []
    tracked_counts: list[int] = []
    last_transform = identity_transform()

    while True:
        ok, current = capture.read()
        if not ok or current is None:
            break

        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        transform = None
        tracked_count = 0
        if previous_points is not None and len(previous_points) >= 3:
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, previous_points, None
            )
            if current_points is not None and status is not None:
                valid = status.reshape(-1).astype(bool)
                valid_previous = previous_points[valid]
                valid_current = current_points[valid]
                tracked_count = len(valid_previous)
                if tracked_count >= 3:
                    # A partial affine transform models translation, rotation,
                    # and uniform scale without introducing shear.
                    transform, _ = cv2.estimateAffinePartial2D(
                        valid_previous, valid_current
                    )

        # Textureless or blurred pairs may not provide enough reliable matches.
        # Reusing the last valid motion estimate avoids a discontinuous jump.
        if transform is None or not np.isfinite(transform).all():
            transform = last_transform.copy()
        else:
            transform = transform.astype(np.float64, copy=False)
            last_transform = transform.copy()

        transforms.append(transform)
        tracked_counts.append(tracked_count)
        previous_gray = current_gray

    if not transforms:
        raise RuntimeError("The input video must contain at least two frames.")
    return transforms, tracked_counts


def transform_parameters(transforms: list[np.ndarray]) -> np.ndarray:
    """Convert affine matrices into x/y translation and rotation parameters."""
    parameters = np.empty((len(transforms), 3), dtype=np.float64)
    for index, transform in enumerate(transforms):
        parameters[index] = (
            transform[0, 2],
            transform[1, 2],
            np.arctan2(transform[1, 0], transform[0, 0]),
        )
    return parameters


def stabilize(
    input_path: Path,
    output_path: Path,
    smoothing_radius: int = DEFAULT_SMOOTHING_RADIUS,
    display: bool = True,
    validate: bool = False,
) -> dict[str, float | int | str]:
    """Run the full two-pass stabilization pipeline and return output metrics."""
    if smoothing_radius < 0:
        raise ValueError("The smoothing radius must be non-negative.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("The input video has invalid dimensions or frame rate.")

    transforms, tracked_counts = estimate_transforms(capture)
    parameters = transform_parameters(transforms)
    trajectory = np.cumsum(parameters, axis=0)
    corrected = parameters + smooth_trajectory(
        trajectory, smoothing_radius
    ) - trajectory

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_width, output_height = 2 * width, height
    if output_width > 1920:
        output_width //= 2
        output_height //= 2
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    written_frames = 0
    for dx, dy, angle in corrected:
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        cosine, sine = np.cos(angle), np.sin(angle)
        transform = np.array(
            [[cosine, -sine, dx], [sine, cosine, dy]], dtype=np.float64
        )
        stabilized = cv2.warpAffine(frame, transform, (width, height))
        stabilized = fix_border(stabilized)
        comparison = cv2.hconcat([frame, stabilized])
        if comparison.shape[1::-1] != (output_width, output_height):
            comparison = cv2.resize(
                comparison, (output_width, output_height)
            )
        writer.write(comparison)
        written_frames += 1

        if display:
            cv2.imshow("Before and After", comparison)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    metrics: dict[str, float | int | str] = {
        "input_frames": len(transforms) + 1,
        "transforms": len(transforms),
        "output_frames": written_frames,
        "width": output_width,
        "height": output_height,
        "mean_tracked_points": float(np.mean(tracked_counts)),
        "output": str(output_path),
    }

    if validate:
        if written_frames != len(transforms):
            raise RuntimeError(
                f"Expected {len(transforms)} output frames, wrote {written_frames}."
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("The output video was not created or is empty.")
        check = cv2.VideoCapture(str(output_path))
        ok, frame = check.read()
        check.release()
        if not ok or frame is None:
            raise RuntimeError("OpenCV could not decode the generated video.")
        if frame.shape[1::-1] != (output_width, output_height):
            raise RuntimeError("The generated video has unexpected dimensions.")
        print(
            "VALIDATION PASSED: "
            f"{written_frames} frames, {output_width}x{output_height}, "
            f"mean tracked points {metrics['mean_tracked_points']:.2f}"
        )

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line controls used by both learners and regression tests."""
    parser = argparse.ArgumentParser(
        description="Stabilize video using point-feature motion estimates."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="video_out.mp4")
    parser.add_argument(
        "--smoothing-radius", type=int, default=DEFAULT_SMOOTHING_RADIUS
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable the preview window."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate generated output."
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line application with concise, actionable errors."""
    args = parse_args()
    try:
        stabilize(
            args.input.resolve(),
            (args.output_dir / args.output_name).resolve(),
            args.smoothing_radius,
            display=not args.no_display,
            validate=args.validate,
        )
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

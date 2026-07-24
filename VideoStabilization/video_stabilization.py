"""
Copyright (c) 2019, Big Vision LLC (Satya Mallick)
https://bigvision.ai  contact@bigvision.ai
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from __future__ import annotations

# Stabilize a video by smoothing feature-based inter-frame camera motion.
import argparse
from pathlib import Path

import cv2
import numpy as np


# Resolve bundled assets from this file, not from the caller's working
# directory, so the example also works when launched by an absolute path.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "video.mp4"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"

# A 101-frame window (50 frames on either side plus the current frame) removes
# high-frequency shake while retaining the slower intentional camera motion.
DEFAULT_SMOOTHING_RADIUS = 50


def moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    """Return an edge-padded moving average with the same length as *curve*."""
    # Reject invalid radii here as well as at the CLI boundary because tests and
    # other Python code can call this reusable function directly.
    if radius < 0:
        raise ValueError("The smoothing radius must be non-negative.")

    # A zero-radius filter is an identity operation; copying prevents callers
    # from accidentally modifying their original array through the result.
    if curve.size == 0 or radius == 0:
        return curve.copy()

    # Edge padding keeps the filter defined at the beginning and end of the clip.
    padded = np.pad(curve, (radius, radius), mode="edge")

    # Normalizing a box kernel by its width makes every output sample the
    # arithmetic mean of the corresponding fixed-size neighborhood.
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / (2 * radius + 1)

    # "valid" removes the padded-only positions and restores the input length.
    return np.convolve(padded, kernel, mode="valid")


def smooth_trajectory(trajectory: np.ndarray, radius: int) -> np.ndarray:
    """Smooth the x translation, y translation, and rotation independently."""
    # Preserve the measured trajectory so that it remains available when the
    # correction is computed later.
    smoothed = trajectory.copy()

    # Each column represents a different physical quantity, so filtering them
    # independently avoids mixing pixels of translation with radians.
    for component in range(trajectory.shape[1]):
        smoothed[:, component] = moving_average(
            trajectory[:, component], radius
        )
    return smoothed


def fix_border(frame: np.ndarray) -> np.ndarray:
    """Scale a stabilized frame around its center to hide moving black borders."""
    # OpenCV reports image shape as rows then columns, hence height precedes width.
    height, width = frame.shape[:2]

    # A small center crop hides most empty wedges introduced by the frame warp.
    transform = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), 0.0, 1.04
    )

    # Keep the original canvas size so every frame matches the VideoWriter.
    return cv2.warpAffine(frame, transform, (width, height))


def identity_transform() -> np.ndarray:
    """Return a 2-by-3 affine identity transform."""
    # Double precision matches estimateAffinePartial2D and limits accumulated
    # rounding error across a long sequence.
    return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64)


def paths_refer_to_same_file(input_path: Path, output_path: Path) -> bool:
    """Return whether two paths name the same destination."""
    # resolve() follows existing symlinks and normalizes relative path segments.
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        return True

    # samefile() additionally detects existing hard links to the input inode.
    if resolved_input.exists() and resolved_output.exists():
        return resolved_input.samefile(resolved_output)
    return False


def estimate_transforms(
    capture: cv2.VideoCapture,
) -> tuple[list[np.ndarray], list[int]]:
    """Estimate partial-affine motion between each pair of decoded frames."""
    # The first decoded frame seeds the pairwise comparison loop.
    ok, previous = capture.read()
    if not ok or previous is None:
        raise RuntimeError("The input video does not contain a readable frame.")

    # Feature detection and optical flow operate on intensity rather than color.
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

    # Keep one transform and one diagnostic point count per adjacent frame pair.
    transforms: list[np.ndarray] = []
    tracked_counts: list[int] = []

    # Identity is a safe starting estimate if the first pair has too little texture.
    last_transform = identity_transform()

    # Decode until the backend reports the end of the readable stream.
    while True:
        ok, current = capture.read()
        if not ok or current is None:
            break

        # Convert only the new frame; the preceding grayscale frame is retained.
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

        # Shi-Tomasi corners provide sparse, well-localized points to track.
        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=3,
        )

        # None marks a frame pair for which no reliable model was estimated.
        transform = None
        tracked_count = 0

        # A partial affine model needs at least three point correspondences.
        if previous_points is not None and len(previous_points) >= 3:
            # Pyramidal Lucas-Kanade follows each corner into the current frame.
            current_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray, current_gray, previous_points, None
            )

            # Some OpenCV backends may return no result for a degenerate pair.
            if current_points is not None and status is not None:
                # The status byte is nonzero only for successfully tracked points.
                valid = status.reshape(-1).astype(bool)
                valid_previous = previous_points[valid]
                valid_current = current_points[valid]
                tracked_count = len(valid_previous)

                # RANSAC inside estimateAffinePartial2D rejects geometric outliers.
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

        # Record the pairwise camera motion and its tracking-quality diagnostic.
        transforms.append(transform)
        tracked_counts.append(tracked_count)

        # Roll the current frame forward for the next adjacent-frame comparison.
        previous_gray = current_gray

    # Stabilization is undefined for a clip with no adjacent frame pair.
    if not transforms:
        raise RuntimeError("The input video must contain at least two frames.")
    return transforms, tracked_counts


def transform_parameters(transforms: list[np.ndarray]) -> np.ndarray:
    """Convert affine matrices into x/y translation and rotation parameters."""
    # Every row stores dx, dy, and angle for one adjacent frame pair.
    parameters = np.empty((len(transforms), 3), dtype=np.float64)

    # Extract the translation column and recover rotation from the 2x2 block.
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
    # Keep programmatic callers subject to the same input validation as the CLI.
    if smoothing_radius < 0:
        raise ValueError("The smoothing radius must be non-negative.")

    # Refuse an alias of the input before VideoWriter could truncate source data.
    if paths_refer_to_same_file(input_path, output_path):
        raise ValueError("The input and output videos must use different paths.")

    # VideoCapture selects an available backend from the supplied path.
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    try:
        # Read output geometry and timing from the decoded stream metadata.
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)

        # Reject unusable metadata before constructing a VideoWriter.
        if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError(
                "The input video has invalid dimensions or frame rate."
            )

        # Pass one measures inter-frame motion for the entire readable clip.
        transforms, tracked_counts = estimate_transforms(capture)
    finally:
        # Always release native decoder resources, including on an exception.
        capture.release()

    # Integrating pairwise motion produces the original camera trajectory.
    parameters = transform_parameters(transforms)
    trajectory = np.cumsum(parameters, axis=0)

    # Add the difference between smooth and measured trajectories to each
    # pairwise transform; this is the motion-correction signal.
    corrected = parameters + smooth_trajectory(
        trajectory, smoothing_radius
    ) - trajectory

    # Create the destination explicitly so output never depends on prior folders.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reopen the path for pass two. This is reliable on backends that cannot
    # seek a compressed stream back to frame zero.
    render_capture = cv2.VideoCapture(str(input_path))
    if not render_capture.isOpened():
        raise RuntimeError(f"Could not reopen input video: {input_path}")

    # The tutorial output places the source and stabilized view side by side.
    output_width, output_height = 2 * width, height

    # Halving very wide comparisons keeps the example's output manageable.
    if output_width > 1920:
        output_width //= 2
        output_height //= 2

    # mp4v is broadly available in OpenCV's common FFmpeg and AVFoundation builds.
    try:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (output_width, output_height),
        )
    except cv2.error:
        render_capture.release()
        raise

    # Fail before the second pass if the selected backend cannot encode the file.
    if not writer.isOpened():
        render_capture.release()
        raise RuntimeError(f"Could not open output video: {output_path}")

    written_frames = 0
    try:
        # There is one corrected transform per adjacent input-frame pair.
        for dx, dy, angle in corrected:
            ok, frame = render_capture.read()
            if not ok or frame is None:
                break

            # Reconstruct a rigid 2x3 transform from translation and rotation.
            cosine, sine = np.cos(angle), np.sin(angle)
            transform = np.array(
                [[cosine, -sine, dx], [sine, cosine, dy]], dtype=np.float64
            )

            # Warp away shake, then crop slightly to conceal empty borders.
            stabilized = cv2.warpAffine(frame, transform, (width, height))
            stabilized = fix_border(stabilized)

            # Concatenate the views so learners can compare them frame by frame.
            comparison = cv2.hconcat([frame, stabilized])

            # Wide source videos use the halved dimensions calculated above.
            if comparison.shape[1::-1] != (output_width, output_height):
                comparison = cv2.resize(
                    comparison, (output_width, output_height)
                )

            # VideoWriter requires every frame to match its configured geometry.
            writer.write(comparison)
            written_frames += 1

            # Headless runs skip all GUI calls inside this block.
            if display:
                cv2.imshow("Before and After", comparison)

                # Escape stops preview; --validate later rejects a partial run.
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        # Flush the encoder and release native resources even after an error.
        render_capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    # Return stable measurements for tests, notebooks, and downstream examples.
    metrics: dict[str, float | int | str] = {
        "input_frames": len(transforms) + 1,
        "transforms": len(transforms),
        "output_frames": written_frames,
        "width": output_width,
        "height": output_height,
        "mean_tracked_points": float(np.mean(tracked_counts)),
        "output": str(output_path),
    }

    # Validation checks semantic output rather than merely successful execution.
    if validate:
        # A complete headless run must write one frame for every correction.
        if written_frames != len(transforms):
            raise RuntimeError(
                f"Expected {len(transforms)} output frames, wrote {written_frames}."
            )

        # A nonempty regular file confirms that the encoder finalized its output.
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("The output video was not created or is empty.")

        # Decode every result frame to catch a short or corrupt encoded stream.
        check = cv2.VideoCapture(str(output_path))
        decoded_frames = 0
        invalid_geometry = False
        try:
            while True:
                ok, frame = check.read()
                if not ok or frame is None:
                    break
                decoded_frames += 1
                if frame.shape[1::-1] != (output_width, output_height):
                    invalid_geometry = True
        finally:
            check.release()

        if decoded_frames == 0:
            raise RuntimeError("OpenCV could not decode the generated video.")

        # Every decoded frame must match the dimensions supplied to VideoWriter.
        if invalid_geometry:
            raise RuntimeError("The generated video has unexpected dimensions.")

        # Accepted writes are not enough: the finalized stream must contain all.
        if decoded_frames != written_frames:
            raise RuntimeError(
                f"Expected {written_frames} decodable output frames, "
                f"found {decoded_frames}."
            )

        # Emit a stable marker and metrics that CIs and readers can inspect.
        print(
            "VALIDATION PASSED: "
            f"{written_frames} frames, {output_width}x{output_height}, "
            f"mean tracked points {metrics['mean_tracked_points']:.2f}"
        )

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line controls used by both learners and regression tests."""
    # argparse supplies consistent --help output and rejects unknown options.
    parser = argparse.ArgumentParser(
        description="Stabilize video using point-feature motion estimates."
    )

    # Path objects keep filesystem handling explicit throughout the program.
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="video_out.mp4")

    # Expose smoothing so learners can observe the stability/crop tradeoff.
    parser.add_argument(
        "--smoothing-radius", type=int, default=DEFAULT_SMOOTHING_RADIUS
    )

    # GUI and validation are opt-in switches suitable for local use and CI.
    parser.add_argument(
        "--no-display", action="store_true", help="Disable the preview window."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate generated output."
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line application with concise, actionable errors."""
    # Keep argument parsing separate so tests can exercise the real CLI contract.
    args = parse_args()
    try:
        # Resolve user paths once, then pass explicit values into the reusable API.
        stabilize(
            args.input.resolve(),
            (args.output_dir / args.output_name).resolve(),
            args.smoothing_radius,
            display=not args.no_display,
            validate=args.validate,
        )
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        # Present expected input/runtime failures without an instructional traceback.
        print(f"ERROR: {error}")
        return 1

    # Zero signals a complete run to shells, CTest wrappers, and notebooks.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

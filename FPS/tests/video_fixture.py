#!/usr/bin/env python3
"""Create and inspect the deterministic video shared by FPS regression tests."""

# Postpone annotation evaluation for the project's Python 3.9 compatibility floor.
from __future__ import annotations

# argparse exposes fixture creation as a separate process for CTest portability.
import argparse
# math checks that the container's FPS metadata is finite and close to the target.
import math
# Path gives every test an explicit, working-directory-independent fixture location.
from pathlib import Path
# Tuple supplies a Python 3.9-compatible fixed-size return annotation.
from typing import Tuple

# OpenCV writes and decodes the same MJPG/AVI fixture used by both example languages.
import cv2
# NumPy constructs small BGR frames without requiring an external media asset.
import numpy as np

# Use a long enough file to test both bounded reads and early end-of-file behavior.
FIXTURE_FRAME_COUNT = 12
# Choose a conventional integer rate that MJPG/AVI backends preserve reliably.
FIXTURE_FPS = 15.0
# Even, small dimensions keep encoding fast and broadly codec-compatible.
FIXTURE_SIZE = (64, 48)
# Allow only minor backend rounding in the reported container metadata.
FPS_TOLERANCE = 0.05


# Generate visibly distinct frames so the fixture represents real decoded video data.
def create_test_video(
    destination: Path,
    frame_count: int = FIXTURE_FRAME_COUNT,
    fps: float = FIXTURE_FPS,
    size: Tuple[int, int] = FIXTURE_SIZE,
) -> None:
    """Write a small MJPG/AVI video and fail clearly when encoding is unavailable."""

    # Create the caller-selected directory before VideoWriter opens the file.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # MJPG in an AVI container is available across more OpenCV builds than MP4 codecs.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # OpenCV expects frame size as (width, height).
    writer = cv2.VideoWriter(str(destination), fourcc, fps, size)
    # A missing encoder must fail the acceptance test rather than silently skip it.
    if not writer.isOpened():
        # Include the intended path to make backend failures easy to diagnose.
        raise RuntimeError(f"Could not create MJPG test video: {destination}")

    # Unpack dimensions once for NumPy's (height, width, channels) array order.
    width, height = size
    # Always release the native encoder, even if constructing a frame raises.
    try:
        # Write the exact number of frames used by stable regression assertions.
        for index in range(frame_count):
            # Fill each frame with an index-dependent BGR color.
            frame = np.full(
                (height, width, 3),
                (index * 17) % 256,
                dtype=np.uint8,
            )
            # Move a bright vertical band so consecutive compressed frames differ.
            band_start = (index * 5) % (width - 4)
            # Use three distinct channels to exercise normal color video decoding.
            frame[:, band_start : band_start + 4] = (255, index * 11 % 256, 0)
            # Submit one correctly sized BGR frame to the encoder.
            writer.write(frame)
    # Flush AVI indexes and release the output file on every path.
    finally:
        # VideoWriter.release() finalizes the container for immediate decoding.
        writer.release()

    # A successful encoder open should leave a nonempty regular file.
    if not destination.is_file() or destination.stat().st_size == 0:
        # Detect backends that failed after opening without surfacing a write status.
        raise RuntimeError(f"Test video was not written: {destination}")


# Decode the completed fixture and return its stable semantic properties.
def inspect_test_video(video_path: Path) -> Tuple[float, int, Tuple[int, int]]:
    """Return ``(reported_fps, decoded_frames, (width, height))``."""

    # Reopen the finalized file through the same VideoCapture API under test.
    capture = cv2.VideoCapture(str(video_path))
    # Fail before property reads when the selected backend cannot decode MJPG/AVI.
    if not capture.isOpened():
        # Make codec availability problems explicit in matrix logs.
        raise RuntimeError(f"Could not open generated test video: {video_path}")

    # Read declared playback FPS before advancing the stream.
    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    # Count successful decodes instead of trusting CAP_PROP_FRAME_COUNT metadata.
    decoded_frames = 0
    # Save the decoded dimensions from real frame data.
    decoded_size = (0, 0)
    # Always release the native decoder and its file handle.
    try:
        # Continue until read() reports the real end of the generated file.
        while True:
            # read() performs the same grab-and-decode work timed by the examples.
            ok, frame = capture.read()
            # A false result marks normal end-of-file for this complete fixture.
            if not ok:
                # Stop without counting the unsuccessful read.
                break
            # A successful read must return a nonempty BGR image.
            if frame is None or frame.size == 0:
                # Catch a backend contract violation before example comparisons.
                raise RuntimeError("Generated video returned an empty frame")
            # Convert NumPy's height/width shape into the documented width/height form.
            current_size = (int(frame.shape[1]), int(frame.shape[0]))
            # Every frame in a normal video must retain the configured dimensions.
            if decoded_size != (0, 0) and current_size != decoded_size:
                # Refuse a malformed fixture with inconsistent geometry.
                raise RuntimeError("Generated video changed dimensions while decoding")
            # Save the first frame's dimensions for the caller.
            decoded_size = current_size
            # Count exactly one successful decoded frame.
            decoded_frames += 1
    # Release the file before temporary-directory cleanup, especially on Windows.
    finally:
        # Explicit cleanup makes the regression portable across file-locking semantics.
        capture.release()

    # Return only values that can remain stable across codec implementations.
    return reported_fps, decoded_frames, decoded_size


# Build the fixture and prove it has the intended semantics before examples consume it.
def create_and_verify_test_video(destination: Path) -> None:
    """Create the standard fixture and validate its stable properties."""

    # Generate a fresh file for each matrix cell.
    create_test_video(destination)
    # Decode the file independently before treating it as test input.
    reported_fps, decoded_frames, decoded_size = inspect_test_video(destination)
    # Require the exact number of frames written by the generator.
    if decoded_frames != FIXTURE_FRAME_COUNT:
        # Avoid blaming either tutorial implementation for a malformed fixture.
        raise RuntimeError(
            f"Expected {FIXTURE_FRAME_COUNT} fixture frames, got {decoded_frames}"
        )
    # Require the exact decoded dimensions selected above.
    if decoded_size != FIXTURE_SIZE:
        # Report both values for actionable backend diagnostics.
        raise RuntimeError(
            f"Expected fixture size {FIXTURE_SIZE}, got {decoded_size}"
        )
    # Require finite FPS metadata close to the requested container rate.
    if not math.isfinite(reported_fps) or not math.isclose(
        reported_fps,
        FIXTURE_FPS,
        abs_tol=FPS_TOLERANCE,
    ):
        # A stable fixture is necessary before comparing Python and C++ metadata.
        raise RuntimeError(
            f"Expected fixture FPS near {FIXTURE_FPS}, got {reported_fps}"
        )


# Parse a destination when the helper is used as CTest's fixture-creation process.
def parse_arguments() -> argparse.Namespace:
    """Return the fixture helper's command-line arguments."""

    # Describe why the generated file exists.
    parser = argparse.ArgumentParser(
        description="Create and preflight the FPS regression video."
    )
    # Require an explicit output so the helper never writes into an arbitrary cwd.
    parser.add_argument("--output", type=Path, required=True)
    # Parse the helper process's real command line.
    return parser.parse_args()


# Create the verified fixture and emit one stable setup marker.
def main() -> int:
    """Run fixture creation as a short-lived standalone process."""

    # Read the absolute destination supplied by the parent regression harness.
    args = parse_arguments()
    # Generate, close, reopen, decode, and close the complete media file.
    create_and_verify_test_video(args.output)
    # Confirm setup before this process releases all OpenCV backend state on exit.
    print("FPS regression fixture created")
    # Signal successful fixture construction.
    return 0


# Execute the helper CLI only when launched directly.
if __name__ == "__main__":
    # End the process after propagating the fixture setup status.
    raise SystemExit(main())

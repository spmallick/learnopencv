#!/usr/bin/env python3
"""Report backend FPS and measure frame-read throughput with OpenCV."""

# Postpone annotation evaluation so the tutorial remains compatible with Python 3.9.
from __future__ import annotations

# argparse provides a documented command-line interface for cameras and video files.
import argparse
# math supplies finite-value checks for backend properties and timing results.
import math
# sys lets runtime failures go to stderr instead of being mixed with measurements.
import sys
# perf_counter is a monotonic clock suited to measuring elapsed durations.
import time
# Callable describes the injectable clock used by deterministic unit tests.
from collections.abc import Callable, Sequence
# NamedTuple gives the measurement fields names while preserving tuple unpacking.
from typing import NamedTuple, Optional, Union

# OpenCV supplies VideoCapture and the CAP_PROP_FPS property used by the example.
import cv2

# Match the signed int accepted by OpenCV's camera-index overload and the C++ example.
CAMERA_INDEX_MIN = -(2**31)
# Values outside this range remain strings so numeric filenames still fail cleanly.
CAMERA_INDEX_MAX = 2**31 - 1


# Keep the related timing values together so callers cannot confuse their order.
class Measurement(NamedTuple):
    """A timed frame-reading result."""

    # timed_fps is frame-read throughput, not necessarily the video's playback FPS.
    timed_fps: float
    # frames_read counts only successful reads inside the timed section.
    frames_read: int
    # elapsed_seconds covers the read-attempt loop, including a terminal EOF probe.
    elapsed_seconds: float


# Convert CLI text into the source type expected by the matching VideoCapture overload.
def parse_video_source(source: str) -> Union[int, str]:
    """Convert an in-range integer source to a camera index."""

    # A complete signed integer such as "0" conventionally selects a camera.
    try:
        # Python integers are unbounded, so parse before applying OpenCV's int range.
        camera_index = int(source)
    # A non-integer value is treated as a filename, URL, or image-sequence pattern.
    except ValueError:
        # Keep file-like sources unchanged so OpenCV can resolve them.
        return source
    # Only values representable by the C++/OpenCV int overload select a camera.
    if CAMERA_INDEX_MIN <= camera_index <= CAMERA_INDEX_MAX:
        # Returning an int selects OpenCV's camera-index VideoCapture constructor.
        return camera_index
    # Treat an oversized numeric token as a string source instead of leaking cv2.error.
    return source


# Read backend-reported FPS while hiding the version-specific unavailable sentinel.
def get_reported_fps(video: cv2.VideoCapture) -> float:
    """Return a positive backend-reported FPS value, or zero when unavailable.

    OpenCV 4 commonly reports an unavailable property as ``0``. OpenCV 5 uses
    ``-1`` for many unavailable VideoCapture properties. Normalizing every
    non-positive or non-finite result keeps the tutorial compatible with both.
    """

    # CAP_PROP_FPS asks the active backend for its frame-rate property value.
    fps = float(video.get(cv2.CAP_PROP_FPS))
    # Only a finite positive value is meaningful as a reported frame rate.
    if math.isfinite(fps) and fps > 0.0:
        # Preserve valid fractional rates such as 29.97.
        return fps
    # Use one stable value for OpenCV 4's 0, OpenCV 5's -1, NaN, or infinity.
    return 0.0


# Time a bounded read-attempt loop with an injectable monotonic clock.
def measure_fps(
    video: cv2.VideoCapture,
    num_frames: int = 120,
    warmup_frames: int = 0,
    clock: Callable[[], float] = time.perf_counter,
) -> Measurement:
    """Read frames and return their timed throughput.

    The numerator uses the actual number of successful reads so a short file is
    never overcounted. Warm-up reads happen before the clock starts. At early
    end-of-file, the elapsed interval also includes the final failed read probe.
    """

    # At least one timed frame must be requested for an FPS calculation.
    if num_frames <= 0:
        # Reject invalid programmatic calls even when argparse is bypassed.
        raise ValueError("num_frames must be greater than zero")
    # A negative warm-up count has no useful interpretation.
    if warmup_frames < 0:
        # Keep the failure close to the invalid caller input.
        raise ValueError("warmup_frames cannot be negative")

    # Discard optional setup frames before measuring camera or decoder steady state.
    for _ in range(warmup_frames):
        # read() grabs and decodes the next frame in one operation.
        ok, _ = video.read()
        # A failed warm-up means the requested timed section cannot be reached.
        if not ok:
            # Distinguish warm-up exhaustion from failure in the timed section.
            raise RuntimeError("Could not read a frame during warm-up")

    # Start timing immediately before the first measured read.
    start = clock()
    # Count successful reads instead of assuming the requested count is available.
    frames_read = 0
    # Stop after the requested maximum even when a camera can stream indefinitely.
    for _ in range(num_frames):
        # Include both grabbing and decoding in the measured operation.
        ok, _ = video.read()
        # False indicates end-of-file, a disconnected camera, or a backend failure.
        if not ok:
            # End the sample without counting the unsuccessful read.
            break
        # Record exactly one successfully returned frame.
        frames_read += 1
    # Stop after the final attempt, including the failed probe that detects EOF.
    elapsed = clock() - start

    # An FPS value cannot be computed when the timed section decoded no frames.
    if frames_read == 0:
        # Report a clear runtime failure instead of returning a misleading zero.
        raise RuntimeError("Could not read any frames from the video source")
    # A duration must be finite and positive before it can be used as a divisor.
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        # This also protects tests or unusual platforms from a broken clock value.
        raise RuntimeError("The elapsed time must be finite and greater than zero")

    # Divide the successful frame count by the measured duration.
    timed_fps = frames_read / elapsed
    # Guard against overflow or any other non-finite arithmetic result.
    if not math.isfinite(timed_fps) or timed_fps <= 0.0:
        # A non-positive throughput cannot describe successful timed reads.
        raise RuntimeError("The timed frame-read rate must be finite and positive")
    # Return named values for printing, validation, and tests.
    return Measurement(timed_fps, frames_read, elapsed)


# Check intrinsic relationships without assuming a particular camera or machine speed.
def validate_measurement(
    reported_fps: float,
    measurement: Measurement,
    requested_frames: int,
) -> None:
    """Raise RuntimeError when a completed measurement is internally inconsistent."""

    # The normalized backend value must always be finite and non-negative.
    if not math.isfinite(reported_fps) or reported_fps < 0.0:
        # Catch a future regression in backend-property normalization.
        raise RuntimeError("Reported FPS validation failed")
    # A successful result must contain at least one frame and never exceed the limit.
    if not 1 <= measurement.frames_read <= requested_frames:
        # Catch frame overcounting as well as impossible empty successes.
        raise RuntimeError("Frame-count validation failed")
    # The raw elapsed duration must retain the invariant enforced during measurement.
    if (
        not math.isfinite(measurement.elapsed_seconds)
        or measurement.elapsed_seconds <= 0.0
    ):
        # Keep validation independent from how the Measurement object was created.
        raise RuntimeError("Elapsed-time validation failed")
    # The published rate must also be finite and positive.
    if not math.isfinite(measurement.timed_fps) or measurement.timed_fps <= 0.0:
        # Reject corrupted or manually constructed measurements.
        raise RuntimeError("Timed-rate validation failed")

    # Recompute the defining relationship from the raw frame count and duration.
    expected_rate = measurement.frames_read / measurement.elapsed_seconds
    # Use a tight relative tolerance to allow only normal floating-point rounding.
    if not math.isclose(measurement.timed_fps, expected_rate, rel_tol=1e-12):
        # A mismatch would mean the displayed rate does not describe the sample.
        raise RuntimeError("Timed-rate calculation validation failed")


# Define the public CLI in one place so help text and tests exercise the same options.
def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    # Describe both backend-property reporting and active throughput measurement.
    parser = argparse.ArgumentParser(
        description="Report backend FPS and measure frame-read throughput."
    )
    # Accept either an integer camera index or a path/URL understood by OpenCV.
    parser.add_argument(
        "--source",
        default="0",
        help="camera index or video path/URL (default: 0)",
    )
    # Retain the tutorial's original 120-frame measurement size as the default.
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="maximum number of frames to time (default: 120)",
    )
    # Let users discard startup frames without including them in the measurement.
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="frames to discard before timing (default: 0)",
    )
    # Offer deterministic intrinsic checks for automated and manual regression runs.
    parser.add_argument(
        "--validate",
        action="store_true",
        help="check measurement invariants and print a success marker",
    )
    # Return the fully configured parser to main() and parser-focused tests.
    return parser


# Open, measure, validate, and release one source while keeping presentation separate.
def run(
    source: Union[int, str],
    num_frames: int = 120,
    warmup_frames: int = 0,
    validate: bool = False,
) -> tuple[float, Measurement]:
    """Measure one source and return its backend FPS property plus timing."""

    # Translate binding-level constructor failures into the tutorial's runtime error.
    try:
        # Construct VideoCapture with either its camera-index or string-source overload.
        video = cv2.VideoCapture(source)
        # Ask the selected backend whether initialization completed successfully.
        source_opened = video.isOpened()
    # OpenCV may raise cv2.error before it can return a closed capture object.
    except (cv2.error, OverflowError) as error:
        # Suppress a binding traceback while retaining the backend diagnostic text.
        raise RuntimeError(f"Could not open video source {source!r}: {error}") from None
    # Fail before reading properties when no backend could open the source.
    if not source_opened:
        # Release any partially initialized backend state before returning the error.
        video.release()
        # repr() makes an empty or whitespace-containing source unambiguous.
        raise RuntimeError(f"Could not open video source {source!r}")

    # Always release cameras and files, including when reading or validation fails.
    try:
        # Query the backend-reported property before reading advances the stream.
        reported_fps = get_reported_fps(video)
        # Perform warm-up and timed reads using the opened capture.
        measurement = measure_fps(video, num_frames, warmup_frames)
        # Run optional invariant checks on the unrounded values.
        if validate:
            # Validation intentionally avoids machine-specific throughput thresholds.
            validate_measurement(reported_fps, measurement, num_frames)
    # VideoCapture owns native resources that should be returned promptly.
    finally:
        # Explicit release is especially important for live cameras and Windows files.
        video.release()

    # Give main() both the backend property and independently timed measurement.
    return reported_fps, measurement


# Parse CLI input, run the example, and format stable human-readable output.
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line program and return a process exit code."""

    # Build a fresh parser so tests can call main() repeatedly without shared state.
    parser = build_argument_parser()
    # Parse the real command line or an explicitly supplied test argument sequence.
    args = parser.parse_args(argv)
    # argparse's type conversion accepts zero, so enforce the semantic lower bound.
    if args.frames <= 0:
        # parser.error prints usage to stderr and exits with the conventional code 2.
        parser.error("--frames must be greater than zero")
    # Warm-up may be zero but cannot be negative.
    if args.warmup_frames < 0:
        # Keep invalid CLI input separate from runtime capture failures.
        parser.error("--warmup-frames cannot be negative")

    # Interpret numeric text as a camera index and everything else as a string source.
    source = parse_video_source(args.source)
    # Convert expected runtime problems into a concise error with exit code 1.
    try:
        # Exercise the same run() path used by programmatic callers and tests.
        reported_fps, measurement = run(
            source,
            num_frames=args.frames,
            warmup_frames=args.warmup_frames,
            validate=args.validate,
        )
    # Invalid sources, exhausted streams, and failed invariants are runtime errors.
    except RuntimeError as error:
        # Send diagnostics to stderr so stdout remains machine-parseable measurements.
        print(f"Error: {error}", file=sys.stderr)
        # Return the conventional nonzero runtime-failure status.
        return 1

    # State the runtime version so matrix logs prove which OpenCV binding executed.
    print(f"OpenCV version: {cv2.__version__}")
    # Print the normalized backend-reported value independently from throughput.
    print(f"Reported frames per second: {reported_fps:.3f}")
    # Echo the requested sample size so a test log is self-describing.
    print(f"Frames requested: {args.frames}")
    # Echo discarded reads because they explain short-file frame counts.
    print(f"Warm-up frames: {args.warmup_frames}")
    # Print the actual successful timed reads rather than the requested maximum.
    print(f"Frames read: {measurement.frames_read}")
    # Preserve enough precision that a very fast file decode does not display as zero.
    print(f"Time taken: {measurement.elapsed_seconds:.9f} seconds")
    # Label this as read throughput so it is not confused with playback FPS.
    print(f"Timed frame-read rate: {measurement.timed_fps:.3f} frames/second")
    # Emit a stable marker only after every requested validation check has passed.
    if args.validate:
        # CTest and manual runs use this exact text as their success criterion.
        print("Validation passed")
    # Signal successful completion to the shell.
    return 0


# Execute main() only when this file is launched as a script.
if __name__ == "__main__":
    # Propagate the returned status as the process exit code.
    raise SystemExit(main())

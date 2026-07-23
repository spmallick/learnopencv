#!/usr/bin/env python3
"""Unit and real-CLI regression tests for frame_rate.py."""

# Postpone annotations to retain Python 3.9 compatibility.
from __future__ import annotations

# math provides NaN and infinity inputs for defensive timing tests.
import math
# os supplies the optional exact-version assertion used by matrix runs.
import os
# subprocess exercises the shipped script exactly as a user invokes it.
import subprocess
# sys supplies the active exact-version interpreter and the project import path.
import sys
# tempfile isolates generated videos and unrelated working directories.
import tempfile
# unittest supplies the standard-library test runner documented by the project.
import unittest
# Path makes every source and fixture location absolute.
from pathlib import Path

# OpenCV supplies the property constant asserted by the metadata unit test.
import cv2

# Discover the project root from this test file instead of the caller's cwd.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Make the example module importable during unittest discovery from any directory.
sys.path.insert(0, str(PROJECT_DIR))

# Import the real Python implementation after establishing its absolute import path.
import frame_rate  # noqa: E402
# Import the shared, preflight-verified media fixture.
from video_fixture import create_and_verify_test_video  # noqa: E402


# Log the active binding and optionally require the matrix cell's exact version.
def setUpModule() -> None:
    """Check the OpenCV version before any Python test executes."""

    # Always leave explicit version evidence in the unittest log.
    print(f"FPS Python tests use OpenCV {cv2.__version__}")
    # Matrix commands set this variable; ordinary compatible-version runs may omit it.
    expected_version = os.environ.get("FPS_EXPECTED_OPENCV_VERSION")
    # Fail before semantic tests if the interpreter belongs to the wrong exact cell.
    if expected_version is not None and cv2.__version__ != expected_version:
        # Include both values so environment mistakes are immediately actionable.
        raise RuntimeError(
            f"Python uses OpenCV {cv2.__version__}, expected {expected_version}"
        )


# Mimic only the VideoCapture methods needed by the pure measurement helpers.
class FakeCapture:
    """A deterministic stand-in for unit tests that should not use real hardware."""

    # Configure the number of successful reads and optional FPS metadata.
    def __init__(self, successful_reads: int, reported_fps: float = 0.0) -> None:
        # Save the total successful reads available across warm-up and timing.
        self.successful_reads = successful_reads
        # Save the backend-property value returned by get().
        self.reported_fps = reported_fps
        # Start with no read attempts.
        self.reads = 0
        # Record the property ID so the test proves CAP_PROP_FPS was requested.
        self.requested_property = None

    # Return success until the configured finite stream is exhausted.
    def read(self):
        # Count every attempt, including the first failed end-of-stream read.
        self.reads += 1
        # Fake frames are unnecessary because measure_fps uses only the success flag.
        return self.reads <= self.successful_reads, None

    # Return deterministic metadata for any requested capture property.
    def get(self, property_id: int) -> float:
        # Save the ID for a precise API assertion.
        self.requested_property = property_id
        # Return the configured OpenCV 4/5-style value.
        return self.reported_fps


# Verify parsing, metadata normalization, timing, and validation without codecs.
class FrameRateUnitTests(unittest.TestCase):
    """Fast tests for each callable behavior in the example module."""

    # Numeric CLI text should select VideoCapture's camera-index overload.
    def test_camera_index_is_parsed_as_integer(self) -> None:
        # Zero is the tutorial's default camera.
        self.assertEqual(frame_rate.parse_video_source("0"), 0)
        # Preserve signed integer behavior from OpenCV's camera API.
        self.assertEqual(frame_rate.parse_video_source("-1"), -1)

    # Non-numeric source text must remain available to file and URL backends.
    def test_file_path_remains_a_string(self) -> None:
        # A normal media filename must not be partially parsed as a camera index.
        self.assertEqual(frame_rate.parse_video_source("12.mp4"), "12.mp4")
        # An integer outside OpenCV's signed-int camera range is also a string source.
        oversized = "9" * 100
        # This aligns Python with C++ stoi overflow and prevents a cv2 traceback.
        self.assertEqual(frame_rate.parse_video_source(oversized), oversized)

    # Positive fractional metadata should pass through unchanged.
    def test_reported_fps_uses_current_property(self) -> None:
        # Simulate a common NTSC-derived frame rate.
        capture = FakeCapture(0, reported_fps=29.97)
        # Preserve the backend value.
        self.assertAlmostEqual(frame_rate.get_reported_fps(capture), 29.97)
        # Prove the current cross-version property constant was used.
        self.assertEqual(capture.requested_property, cv2.CAP_PROP_FPS)

    # OpenCV 4 and 5 use different unavailable-property sentinels.
    def test_invalid_reported_fps_is_normalized_to_zero(self) -> None:
        # Cover OpenCV 4's 0, OpenCV 5's -1, and defensive non-finite cases.
        for value in (0.0, -1.0, float("nan"), float("inf")):
            # Identify the failing sentinel if a future change regresses.
            with self.subTest(value=value):
                # All unavailable or invalid values share one documented output.
                self.assertEqual(
                    frame_rate.get_reported_fps(FakeCapture(0, value)),
                    0.0,
                )

    # Early end-of-file must use the successful count rather than the request.
    def test_measurement_uses_successfully_read_frames(self) -> None:
        # Supply four frames even though the caller requests ten.
        capture = FakeCapture(successful_reads=4)
        # Make the measured duration exactly two seconds.
        clock = iter((10.0, 12.0)).__next__
        # Run the real measurement loop.
        result = frame_rate.measure_fps(capture, num_frames=10, clock=clock)
        # Require the four real successes.
        self.assertEqual(result.frames_read, 4)
        # Preserve the injected duration.
        self.assertEqual(result.elapsed_seconds, 2.0)
        # Four frames in two seconds is exactly two frames per second.
        self.assertEqual(result.timed_fps, 2.0)

    # Warm-up reads must happen before the clock and not enter the frame count.
    def test_warmup_frames_are_not_in_measurement(self) -> None:
        # Provide exactly two warm-up and three timed frames.
        capture = FakeCapture(successful_reads=5)
        # Make the timed section exactly one second.
        clock = iter((3.0, 4.0)).__next__
        # Request the configured warm-up and timed samples.
        result = frame_rate.measure_fps(
            capture,
            num_frames=3,
            warmup_frames=2,
            clock=clock,
        )
        # Count only the three reads after timing starts.
        self.assertEqual(result.frames_read, 3)
        # Confirm all five source frames were consumed.
        self.assertEqual(capture.reads, 5)

    # Empty timed samples should fail instead of returning a zero rate.
    def test_empty_source_raises_clear_error(self) -> None:
        # Match the user-facing failure text.
        with self.assertRaisesRegex(RuntimeError, "Could not read any frames"):
            # The fake source fails on its first timed read.
            frame_rate.measure_fps(
                FakeCapture(0),
                num_frames=1,
                clock=iter((0.0, 1.0)).__next__,
            )

    # A source exhausted during warm-up should name that phase.
    def test_failed_warmup_raises_clear_error(self) -> None:
        # Require the phase-specific diagnostic.
        with self.assertRaisesRegex(RuntimeError, "during warm-up"):
            # One available frame cannot satisfy two warm-up reads.
            frame_rate.measure_fps(
                FakeCapture(1),
                num_frames=1,
                warmup_frames=2,
                clock=iter((0.0, 1.0)).__next__,
            )

    # Zero, backward, and non-finite clock results are invalid divisors.
    def test_invalid_elapsed_time_is_rejected(self) -> None:
        # Cover equal timestamps, a backward clock, NaN, and positive infinity.
        for start, end in (
            (2.0, 2.0),
            (2.0, 1.0),
            (0.0, math.nan),
            (0.0, math.inf),
        ):
            # Preserve the problematic values in a subtest label.
            with self.subTest(start=start, end=end):
                # Require a clear finite-positive-duration error.
                with self.assertRaisesRegex(RuntimeError, "elapsed time"):
                    # One frame succeeds before the injected clock result is checked.
                    frame_rate.measure_fps(
                        FakeCapture(1),
                        num_frames=1,
                        clock=iter((start, end)).__next__,
                    )

    # Programmatic callers receive ValueError for invalid count arguments.
    def test_invalid_frame_counts_are_rejected(self) -> None:
        # Zero timed frames cannot produce an FPS measurement.
        with self.assertRaises(ValueError):
            # Bypass argparse to test the function's own contract.
            frame_rate.measure_fps(FakeCapture(1), num_frames=0)
        # Negative warm-up counts are nonsensical.
        with self.assertRaises(ValueError):
            # Bypass argparse to test the second function precondition.
            frame_rate.measure_fps(FakeCapture(1), warmup_frames=-1)

    # Validation should accept exactly consistent raw measurement values.
    def test_intrinsic_validation_accepts_consistent_measurement(self) -> None:
        # Construct a rate of two frames per second from four frames in two seconds.
        measurement = frame_rate.Measurement(2.0, 4, 2.0)
        # No exception means every machine-independent invariant passed.
        frame_rate.validate_measurement(15.0, measurement, requested_frames=10)

    # Validation should reject a rate unrelated to its raw count and duration.
    def test_intrinsic_validation_rejects_inconsistent_rate(self) -> None:
        # Deliberately claim three FPS for a two-FPS sample.
        measurement = frame_rate.Measurement(3.0, 4, 2.0)
        # Require the calculation-specific diagnostic.
        with self.assertRaisesRegex(RuntimeError, "calculation"):
            # Validate the deliberately corrupted result.
            frame_rate.validate_measurement(15.0, measurement, 10)


# Exercise the shipped Python entry point with real OpenCV video I/O.
class FrameRatePythonCliTests(unittest.TestCase):
    """Subprocess tests that detect cwd, argparse, and resource-handling regressions."""

    # Generate one preflight-verified video for each isolated test.
    def setUp(self) -> None:
        # Own the temporary directory until tearDown() completes.
        self.temporary_directory = tempfile.TemporaryDirectory()
        # Convert its name to a Path for explicit child locations.
        self.temp_path = Path(self.temporary_directory.name)
        # Place the source outside the child's unrelated working directory.
        self.video_path = self.temp_path / "fixture.avi"
        # Refuse to run CLI assertions against an unverified media file.
        create_and_verify_test_video(self.video_path)
        # Create an empty cwd that contains none of the project files.
        self.unrelated_cwd = self.temp_path / "unrelated-cwd"
        # Materialize that cwd before passing it to subprocess.
        self.unrelated_cwd.mkdir()

    # Release temporary videos and directories after every subprocess test.
    def tearDown(self) -> None:
        # TemporaryDirectory performs recursive cleanup after OpenCV processes exit.
        self.temporary_directory.cleanup()

    # Build the common real-script command with an absolute source path.
    def command(self, *extra_arguments: str) -> list[str]:
        # Use the active exact-version interpreter from the validation environment.
        return [
            sys.executable,
            str(PROJECT_DIR / "frame_rate.py"),
            "--source",
            str(self.video_path),
            *extra_arguments,
        ]

    # Short-file behavior, warm-up exclusion, and validation must work end to end.
    def test_cli_from_unrelated_directory(self) -> None:
        # Request more frames than remain after three warm-up reads.
        completed = subprocess.run(
            self.command(
                "--warmup-frames",
                "3",
                "--frames",
                "20",
                "--validate",
            ),
            cwd=self.unrelated_cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        # Print child diagnostics through unittest only if this assertion fails.
        self.assertEqual(completed.returncode, 0, completed.stderr)
        # Twelve fixture frames minus three warm-up frames leaves exactly nine.
        self.assertIn("Frames read: 9", completed.stdout)
        # The marker proves validation ran after the raw measurement checks.
        self.assertIn("Validation passed", completed.stdout)
        # Permit nonfatal backend fallback chatter but never a Python traceback.
        self.assertNotIn("Traceback", completed.stderr)
        # The example has no output files and must leave an unrelated cwd untouched.
        self.assertEqual(list(self.unrelated_cwd.iterdir()), [])

    # Invalid CLI counts should use argparse's status and avoid a traceback.
    def test_cli_rejects_zero_frames(self) -> None:
        # Invoke the real parser with an invalid semantic lower bound.
        completed = subprocess.run(
            self.command("--frames", "0"),
            cwd=self.unrelated_cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        # argparse conventionally returns status two for command misuse.
        self.assertEqual(completed.returncode, 2)
        # The precise option rule should be visible to the user.
        self.assertIn("--frames must be greater than zero", completed.stderr)
        # Parser errors should never emit a Python traceback.
        self.assertNotIn("Traceback", completed.stderr)
        # Failed invocations cannot claim validation success.
        self.assertNotIn("Validation passed", completed.stdout)

    # A nonexistent media path should be a concise runtime failure.
    def test_cli_reports_missing_source(self) -> None:
        # Point the real CLI at an absolute path that was never created.
        missing_path = self.temp_path / "missing.avi"
        # Run without allowing subprocess to raise on the expected failure.
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "frame_rate.py"),
                "--source",
                str(missing_path),
            ],
            cwd=self.unrelated_cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        # Source-open failures use the runtime status one.
        self.assertEqual(completed.returncode, 1)
        # OpenCV may add backend warnings, so assert only the tutorial diagnostic.
        self.assertIn("Could not open video source", completed.stderr)
        # The wrapper should not leak a traceback.
        self.assertNotIn("Traceback", completed.stderr)

    # Oversized numeric text should fail as a string source without a binding traceback.
    def test_cli_handles_oversized_numeric_source(self) -> None:
        # Use a value far outside the signed-int range accepted by VideoCapture.
        oversized = "9" * 100
        # Exercise the real script from the unrelated working directory.
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "frame_rate.py"),
                "--source",
                oversized,
            ],
            cwd=self.unrelated_cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        # The oversized token is a nonexistent string source, hence runtime status one.
        self.assertEqual(completed.returncode, 1)
        # The tutorial should still provide its concise source-open diagnostic.
        self.assertIn("Could not open video source", completed.stderr)
        # Binding exceptions must not escape as a traceback.
        self.assertNotIn("Traceback", completed.stderr)


# Use unittest's normal CLI when this file is launched directly.
if __name__ == "__main__":
    # Run every test class above and propagate its result as the process status.
    unittest.main()

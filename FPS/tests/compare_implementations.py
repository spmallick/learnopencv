#!/usr/bin/env python3
"""Run Python and C++ FPS CLIs on one video and compare stable results."""

# Postpone annotations for the project's Python 3.9 compatibility floor.
from __future__ import annotations

# argparse documents the absolute paths supplied by CTest.
import argparse
# math validates parsed floating-point output without comparing machine speed.
import math
# re extracts stable named metrics from each human-readable CLI report.
import re
# subprocess executes the two real tutorial entry points.
import subprocess
# sys identifies the exact Python interpreter selected during CMake configuration.
import sys
# tempfile supplies a fresh fixture and unrelated cwd for each matrix cell.
import tempfile
# NamedTuple gives parsed report fields readable names.
from typing import NamedTuple
# Path prevents any dependency on CTest's working directory.
from pathlib import Path

# OpenCV creates and preflights the shared fixture in this exact Python environment.
import cv2

# Import constants and generation logic colocated with this harness.
from video_fixture import (
    FIXTURE_FPS,
    FPS_TOLERANCE,
)


# Store only the output values that are safe to compare across languages.
class Report(NamedTuple):
    """Parsed output from one successful tutorial process."""

    # version proves the expected OpenCV matrix cell actually executed.
    version: str
    # reported_fps is the backend property and should agree for one shared container.
    reported_fps: float
    # frames_read is deterministic for the finite fixture and requested bounds.
    frames_read: int
    # elapsed_seconds is checked only for positivity, never equality.
    elapsed_seconds: float
    # timed_fps is checked only for positivity, never cross-language performance.
    timed_fps: float


# Parse the three absolute inputs that make the CTest invocation reproducible.
def parse_arguments() -> argparse.Namespace:
    """Return validated harness arguments."""

    # Describe the shared-source comparison performed by this test.
    parser = argparse.ArgumentParser(
        description="Compare stable Python and C++ FPS example results."
    )
    # Point at the shipped Python script rather than importing a duplicate workflow.
    parser.add_argument("--python-script", type=Path, required=True)
    # Point at the executable produced in the current fresh CMake build.
    parser.add_argument("--cpp-executable", type=Path, required=True)
    # Require the runtime/binary version to match CMake's resolved OpenCV package.
    parser.add_argument("--expected-opencv-version", required=True)
    # Parse the real CTest command line.
    return parser.parse_args()


# Extract one required regex group or fail with the complete child output.
def require_match(pattern: str, output: str, label: str) -> str:
    """Return one matched group from output."""

    # Search in multiline output without assuming any line ordering beyond its label.
    match = re.search(pattern, output, flags=re.MULTILINE)
    # Missing fields indicate an incompatible or incomplete CLI result.
    if match is None:
        # Include the process label and stdout for immediate diagnosis.
        raise RuntimeError(f"{label} output missed {pattern!r}:\n{output}")
    # Return the first capturing group for numeric or text conversion.
    return match.group(1)


# Convert one successful process report into typed stable and sanity-check values.
def parse_report(
    output: str,
    label: str,
    expect_validation: bool,
) -> Report:
    """Parse and validate one CLI's measurement report."""

    # Read the explicit build/runtime version line.
    version = require_match(r"^OpenCV version: (.+)$", output, label)
    # Parse the normalized backend-reported FPS property as a floating-point value.
    reported_fps = float(
        require_match(
            r"^Reported frames per second: ([0-9.+-]+)$",
            output,
            label,
        )
    )
    # Parse the exact successful timed-frame count.
    frames_read = int(require_match(r"^Frames read: ([0-9]+)$", output, label))
    # Parse the raw duration printed with enough precision to remain nonzero.
    elapsed_seconds = float(
        require_match(r"^Time taken: ([0-9.eE+-]+) seconds$", output, label)
    )
    # Parse throughput only for a finite-positive sanity check.
    timed_fps = float(
        require_match(
            r"^Timed frame-read rate: ([0-9.eE+-]+) frames/second$",
            output,
            label,
        )
    )
    # Record whether the optional validation marker appears.
    validation_marker_present = "Validation passed" in output
    # A validated child must emit its marker after intrinsic raw-value checks.
    if expect_validation and not validation_marker_present:
        # Reject a run that measured but did not execute the requested checks.
        raise RuntimeError(f"{label} did not report validation success:\n{output}")
    # A normal run without --validate must not print a misleading success marker.
    if not expect_validation and validation_marker_present:
        # Catch an accidentally unconditional marker.
        raise RuntimeError(f"{label} reported unrequested validation:\n{output}")
    # Timing values vary by implementation and platform but must remain meaningful.
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        # Diagnose a rounded-to-zero or invalid duration.
        raise RuntimeError(f"{label} reported invalid elapsed time: {elapsed_seconds}")
    # Throughput likewise must be finite and positive without needing a speed target.
    if not math.isfinite(timed_fps) or timed_fps <= 0.0:
        # Avoid turning performance into a compatibility assertion.
        raise RuntimeError(f"{label} reported invalid timed FPS: {timed_fps}")
    # Return the parsed report for stable cross-language comparisons.
    return Report(
        version,
        reported_fps,
        frames_read,
        elapsed_seconds,
        timed_fps,
    )


# Launch one real entry point from an unrelated cwd and require clean success.
def run_command(
    command: list[str],
    cwd: Path,
    label: str,
    expect_validation: bool = True,
) -> Report:
    """Execute one implementation and return its parsed report."""

    # Capture both streams so diagnostics and machine-readable output stay distinct.
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    # Every successful measurement, with or without validation, returns status zero.
    if completed.returncode != 0:
        # Include both streams because OpenCV backends may write their own diagnostics.
        raise RuntimeError(
            f"{label} exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    # OpenCV may emit nonfatal backend-fallback warnings even after a valid decode.
    if "Traceback" in completed.stderr or "terminate called" in completed.stderr:
        # Reject language/runtime crashes while tolerating platform backend chatter.
        raise RuntimeError(f"{label} leaked an exception:\n{completed.stderr}")
    # Parse all required output fields and the expected validation-marker state.
    return parse_report(completed.stdout, label, expect_validation)


# Require one CLI misuse or runtime failure without accepting a false success marker.
def require_failure(
    command: list[str],
    cwd: Path,
    label: str,
    expected_status: int,
    expected_text: str,
) -> None:
    """Check one real CLI failure path."""

    # Run the child without converting its expected nonzero status into an exception.
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    # Distinguish argument misuse (2) from capture/runtime failure (1).
    if completed.returncode != expected_status:
        # Include both streams so a platform-specific backend failure is visible.
        raise RuntimeError(
            f"{label} exited {completed.returncode}, expected {expected_status}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    # Match a stable diagnostic fragment while allowing backend-added warnings.
    if expected_text not in completed.stderr:
        # Make a misleading or missing user-facing error actionable.
        raise RuntimeError(
            f"{label} stderr missed {expected_text!r}:\n{completed.stderr}"
        )
    # Tutorial wrappers should never expose a language traceback or uncaught exception.
    if "Traceback" in completed.stderr or "terminate called" in completed.stderr:
        # Reject a crash even when it happens to return the expected status.
        raise RuntimeError(f"{label} leaked an exception:\n{completed.stderr}")
    # A failed command must never claim that validation succeeded.
    if "Validation passed" in completed.stdout:
        # Protect CTest from accepting a stale or prematurely printed marker.
        raise RuntimeError(f"{label} printed a false validation marker")


# Require --help to return normally without trying to open the default camera.
def require_help(command: list[str], cwd: Path, label: str) -> None:
    """Check one real CLI help path."""

    # Add junk after help to prove both parsers stop immediately on the help action.
    completed = subprocess.run(
        [*command, "--help", "--unknown-after-help"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    # Help is a successful request in both implementations.
    if completed.returncode != 0:
        # Include the diagnostic produced by an unexpected failure.
        raise RuntimeError(
            f"{label} help exited {completed.returncode}:\n{completed.stderr}"
        )
    # Both parsers expose a usage line without opening a source.
    if "usage:" not in completed.stdout.lower():
        # Reject empty or accidentally removed help output.
        raise RuntimeError(f"{label} help omitted usage:\n{completed.stdout}")
    # A help-only invocation should have no runtime or backend diagnostic.
    if completed.stderr:
        # This can reveal an accidental default-camera open.
        raise RuntimeError(f"{label} help wrote to stderr:\n{completed.stderr}")


# Create the video in a short-lived process so every native encoder handle is gone.
def create_fixture_in_subprocess(video_path: Path) -> None:
    """Generate and preflight the shared media fixture."""

    # Resolve the helper colocated with this harness.
    fixture_script = Path(__file__).resolve().with_name("video_fixture.py")
    # Let process exit tear down backend-global codec state before C++ opens the file.
    completed = subprocess.run(
        [sys.executable, str(fixture_script), "--output", str(video_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    # Fixture construction is a required acceptance step, never an optional skip.
    if completed.returncode != 0:
        # Preserve the helper's complete output for codec/backend diagnosis.
        raise RuntimeError(
            f"Fixture creation exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    # Require the marker printed only after encode and decode preflight succeed.
    if "FPS regression fixture created" not in completed.stdout:
        # Reject an incomplete helper execution even if it returned zero.
        raise RuntimeError(f"Fixture setup marker missing:\n{completed.stdout}")
    # Backend fallback warnings are nonfatal when encode/decode preflight succeeded.
    if "Traceback" in completed.stderr or "terminate called" in completed.stderr:
        # Reject a language/runtime crash while allowing platform codec chatter.
        raise RuntimeError(f"Fixture creation leaked an exception:\n{completed.stderr}")


# Compare one bounded or end-of-file case using the same absolute source path.
def compare_case(
    python_script: Path,
    cpp_executable: Path,
    video_path: Path,
    cwd: Path,
    expected_version: str,
    warmup_frames: int,
    requested_frames: int,
    expected_frames: int,
    validate: bool = True,
) -> None:
    """Run both CLIs and compare stable metadata and frame counts."""

    # Keep the option sequence identical for Python and C++.
    common_arguments = [
        "--source",
        str(video_path),
        "--warmup-frames",
        str(warmup_frames),
        "--frames",
        str(requested_frames),
    ]
    # Add validation only for cases intended to exercise the invariant checks.
    if validate:
        # Keep the flag position identical for both implementations.
        common_arguments.append("--validate")
    # Use the selected exact-version Python interpreter for the Python script.
    python_report = run_command(
        [sys.executable, str(python_script), *common_arguments],
        cwd,
        "Python",
        expect_validation=validate,
    )
    # Run the freshly built C++ executable on the identical fixture and options.
    cpp_report = run_command(
        [str(cpp_executable), *common_arguments],
        cwd,
        "C++",
        expect_validation=validate,
    )

    # Prove both entry points used the OpenCV version selected by this CMake build.
    if python_report.version != expected_version:
        # Catch a Python environment from the wrong exact matrix cell.
        raise RuntimeError(
            f"Python used OpenCV {python_report.version}, expected {expected_version}"
        )
    # The linked core library's runtime string must match find_package(OpenCV).
    if cpp_report.version != expected_version:
        # Catch a runtime library accidentally resolved from a different installation.
        raise RuntimeError(
            f"C++ used OpenCV {cpp_report.version}, expected {expected_version}"
        )
    # Both finite-source loops must count the exact expected successful reads.
    if python_report.frames_read != expected_frames:
        # Report the full Python count mismatch.
        raise RuntimeError(
            f"Python read {python_report.frames_read}, expected {expected_frames}"
        )
    # Apply the same exact count assertion to C++.
    if cpp_report.frames_read != expected_frames:
        # Report the full C++ count mismatch.
        raise RuntimeError(
            f"C++ read {cpp_report.frames_read}, expected {expected_frames}"
        )
    # Each backend should expose the generated container's declared 15 FPS metadata.
    if not math.isclose(
        python_report.reported_fps,
        FIXTURE_FPS,
        abs_tol=FPS_TOLERANCE,
    ):
        # Do not accept a malformed or differently interpreted Python source.
        raise RuntimeError(
            f"Python reported {python_report.reported_fps} FPS, "
            f"expected near {FIXTURE_FPS}"
        )
    # Check the C++ backend against the same container metadata tolerance.
    if not math.isclose(
        cpp_report.reported_fps,
        FIXTURE_FPS,
        abs_tol=FPS_TOLERANCE,
    ):
        # Keep the diagnostic language-specific.
        raise RuntimeError(
            f"C++ reported {cpp_report.reported_fps} FPS, "
            f"expected near {FIXTURE_FPS}"
        )
    # The two bindings should interpret one container's stable metadata equally.
    if not math.isclose(
        python_report.reported_fps,
        cpp_report.reported_fps,
        abs_tol=FPS_TOLERANCE,
    ):
        # Never extend this comparison to elapsed time or throughput.
        raise RuntimeError(
            "Python/C++ reported FPS mismatch: "
            f"{python_report.reported_fps} vs {cpp_report.reported_fps}"
        )


# Create one fixture, exercise success/error modes, and print the CTest marker.
def main() -> int:
    """Run the complete cross-language regression."""

    # Parse CMake-provided paths and its resolved exact OpenCV version.
    args = parse_arguments()
    # Resolve paths now so child invocations remain independent of cwd.
    python_script = args.python_script.resolve()
    # Resolve the generated executable path for the same reason.
    cpp_executable = args.cpp_executable.resolve()
    # Fail clearly if CMake or packaging omitted either real entry point.
    if not python_script.is_file():
        # Name the missing Python path.
        raise RuntimeError(f"Python example not found: {python_script}")
    # The C++ target must exist and be executable before CTest begins.
    if not cpp_executable.is_file():
        # Name the missing binary path.
        raise RuntimeError(f"C++ example not found: {cpp_executable}")
    # The harness itself must import the expected exact-version cv2 binding.
    if cv2.__version__ != args.expected_opencv_version:
        # Prevent a mixed-version cross-language test from being reported as valid.
        raise RuntimeError(
            f"Test interpreter uses OpenCV {cv2.__version__}, "
            f"expected {args.expected_opencv_version}"
        )

    # Remove all generated media automatically after both child processes exit.
    with tempfile.TemporaryDirectory() as directory:
        # Treat the temporary root as an explicit absolute path.
        temp_path = Path(directory)
        # Keep the input outside the child processes' working directory.
        video_path = temp_path / "fixture.avi"
        # Generate and decode-preflight exactly twelve 15-FPS frames in a child.
        create_fixture_in_subprocess(video_path)
        # Create an empty directory that contains no source or build files.
        unrelated_cwd = temp_path / "unrelated-cwd"
        # Materialize it before subprocess uses it.
        unrelated_cwd.mkdir()

        # Exercise warm-up exclusion plus early end-of-file: 12 - 3 = 9.
        compare_case(
            python_script,
            cpp_executable,
            video_path,
            unrelated_cwd,
            args.expected_opencv_version,
            warmup_frames=3,
            requested_frames=20,
            expected_frames=9,
        )
        # Exercise a request-limited sample that stops before end-of-file.
        compare_case(
            python_script,
            cpp_executable,
            video_path,
            unrelated_cwd,
            args.expected_opencv_version,
            warmup_frames=2,
            requested_frames=4,
            expected_frames=4,
        )
        # Exercise the normal successful mode without validation or its marker.
        compare_case(
            python_script,
            cpp_executable,
            video_path,
            unrelated_cwd,
            args.expected_opencv_version,
            warmup_frames=1,
            requested_frames=2,
            expected_frames=2,
            validate=False,
        )
        # Build reusable command prefixes for aligned parser and runtime error checks.
        python_command = [sys.executable, str(python_script)]
        # The C++ command prefix contains only the freshly built executable.
        cpp_command = [str(cpp_executable)]
        # Help must succeed without touching camera zero.
        require_help(python_command, unrelated_cwd, "Python")
        # Apply the identical help expectation to C++.
        require_help(cpp_command, unrelated_cwd, "C++")

        # Check invalid frame-count semantics and exit code two in both languages.
        for label, command in (
            ("Python", python_command),
            ("C++", cpp_command),
        ):
            # Zero timed frames is valid integer syntax but invalid program input.
            require_failure(
                [*command, "--frames", "0"],
                unrelated_cwd,
                label,
                expected_status=2,
                expected_text="--frames",
            )
            # An unknown option should be diagnosed before looking for a value.
            require_failure(
                [*command, "--unknown"],
                unrelated_cwd,
                label,
                expected_status=2,
                expected_text="--unknown",
            )
            # A following option token cannot silently become --source's value.
            require_failure(
                [*command, "--source", "--frames", "10"],
                unrelated_cwd,
                label,
                expected_status=2,
                expected_text="--source",
            )
            # Exhausting all frames exactly during warm-up leaves no timed sample.
            require_failure(
                [
                    *command,
                    "--source",
                    str(video_path),
                    "--warmup-frames",
                    "12",
                    "--frames",
                    "1",
                ],
                unrelated_cwd,
                label,
                expected_status=1,
                expected_text="Could not read any frames",
            )
            # Requesting more warm-up frames than exist fails in the warm-up phase.
            require_failure(
                [
                    *command,
                    "--source",
                    str(video_path),
                    "--warmup-frames",
                    "13",
                    "--frames",
                    "1",
                ],
                unrelated_cwd,
                label,
                expected_status=1,
                expected_text="during warm-up",
            )
            # A nonexistent absolute file is a runtime source-open failure.
            require_failure(
                [*command, "--source", str(temp_path / "missing.avi")],
                unrelated_cwd,
                label,
                expected_status=1,
                expected_text="Could not open video source",
            )
            # Oversized integers become string sources and must not crash a binding.
            require_failure(
                [*command, "--source", "9" * 100],
                unrelated_cwd,
                label,
                expected_status=1,
                expected_text="Could not open video source",
            )
        # Neither example displays frames or writes outputs into the caller's cwd.
        if list(unrelated_cwd.iterdir()):
            # Enumerate unexpected artifacts for a precise failure.
            raise RuntimeError(
                f"Examples created unexpected files: {list(unrelated_cwd.iterdir())}"
            )

    # Summarize the exact matrix cell before the stable success marker.
    print(f"Validated Python and C++ with OpenCV {args.expected_opencv_version}")
    # CTest's PASS_REGULAR_EXPRESSION requires this exact final text.
    print("Cross-language FPS validation passed")
    # Return a successful process status.
    return 0


# Run the harness only when CTest or a user launches this file directly.
if __name__ == "__main__":
    # Propagate the result as the process exit status.
    raise SystemExit(main())

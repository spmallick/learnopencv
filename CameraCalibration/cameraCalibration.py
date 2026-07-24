#!/usr/bin/env python3
"""Calibrate a camera from checkerboard images with OpenCV 4.14 or 5.0."""

from __future__ import annotations

# argparse gives the teaching script discoverable command-line options.
import argparse
# sys supplies the process argument list and standard error stream.
import sys
# Path anchors bundled assets to this file instead of the caller's directory.
from pathlib import Path
# Sequence describes either a real or test-provided argument list.
from typing import Sequence

# Shared helpers keep both tutorial entry points numerically identical.
from calibration_utils import calibrate_images, print_calibration, validate_calibration


# Resolve the 41 bundled images even when the script starts elsewhere.
PROJECT_DIR = Path(__file__).resolve().parent
# Quote this wildcard at a shell when overriding it so Python, not the shell,
# controls the complete input set.
DEFAULT_IMAGE_PATTERN = str(PROJECT_DIR / "images" / "*.jpg")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options, optionally from a test-provided sequence."""

    # Use the module documentation as the command's short help description.
    parser = argparse.ArgumentParser(description=__doc__)
    # A custom glob lets readers calibrate their own same-sized checkerboards.
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGE_PATTERN,
        help="Calibration image glob (default: bundled images)",
    )
    # Headless mode avoids all window-system calls on servers and in CI.
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip the interactive checkerboard preview",
    )
    # Validation turns silent numeric assumptions into explicit failures.
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check version-independent calibration invariants",
    )
    # argparse uses sys.argv when arguments is None.
    return parser.parse_args(arguments)


def run(options: argparse.Namespace) -> None:
    """Execute calibration with already parsed options."""

    # The helper detects corners, refines them, and estimates camera parameters.
    calibration = calibrate_images(
        options.images,
        display=not options.no_display,
    )
    # Print coverage, errors, intrinsics, and distortion for the lesson.
    print_calibration(calibration)
    # Keep validation opt-in for custom datasets with different quality.
    if options.validate:
        validate_calibration(calibration)
        # Tests require an unambiguous marker after every check succeeds.
        print("Validation passed")


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and convert operational failures into a clean exit status."""

    try:
        # Parse first so argparse can retain its conventional exit code and help.
        options = parse_args(arguments)
        # Execute the real tutorial path rather than a test-only implementation.
        run(options)
    except Exception as error:
        # OpenCV and filesystem errors become a concise command-line diagnostic.
        print(f"Error: {error}", file=sys.stderr)
        return 1
    # Zero tells shells and CTest that all requested work completed.
    return 0


if __name__ == "__main__":
    # SystemExit propagates main's explicit success or failure status.
    raise SystemExit(main())

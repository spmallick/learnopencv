#!/usr/bin/env python3
"""Calibrate a camera and compare two OpenCV undistortion workflows."""

from __future__ import annotations

# argparse documents the inputs, outputs, headless mode, and validation mode.
import argparse
# sys supplies standard error and the final process status.
import sys
# Path makes input and output locations explicit and platform independent.
from pathlib import Path
# Sequence permits direct testing without modifying process-global arguments.
from typing import Sequence

# Shared calibration logic prevents the two lessons from drifting apart.
from calibration_utils import (
    calibrate_images,
    print_calibration,
    undistort_sample,
    validate_calibration,
)


# Locate bundled data relative to the source file, not the current directory.
PROJECT_DIR = Path(__file__).resolve().parent
# The wildcard selects all 41 tracked checkerboard images.
DEFAULT_IMAGE_PATTERN = str(PROJECT_DIR / "images" / "*.jpg")
# The two documented methods should remain below one mean intensity level; the
# tighter 0.1 threshold passed on both exact reference versions.
MAXIMUM_METHOD_DIFFERENCE = 0.1


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options, optionally from a test-provided sequence."""

    # Use the module documentation as the command's concise help text.
    parser = argparse.ArgumentParser(description=__doc__)
    # A quoted glob can select an alternate same-sized calibration dataset.
    parser.add_argument(
        "--images",
        default=DEFAULT_IMAGE_PATTERN,
        help="Calibration image glob (default: bundled images)",
    )
    # By default the first sorted calibration image demonstrates undistortion.
    parser.add_argument(
        "--sample",
        type=Path,
        help="Image to undistort (default: first calibration image)",
    )
    # Keep generated files out of the source tree during normal validation.
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory (default: current directory)",
    )
    # Headless mode avoids imshow and waitKey on servers and in CI.
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip interactive checkerboard and undistortion windows",
    )
    # Validation checks both calibration quality and method agreement.
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check calibration and undistortion invariants",
    )
    # argparse reads sys.argv when arguments is None.
    return parser.parse_args(arguments)


def run(options: argparse.Namespace) -> None:
    """Execute calibration and both undistortion methods."""

    # Calibrate from the complete selected checkerboard set.
    calibration = calibrate_images(
        options.images,
        display=not options.no_display,
    )
    # Report the same reusable camera parameters as the calibration-only lesson.
    print_calibration(calibration)
    # A user-supplied sample wins; otherwise sorting makes the first match stable.
    sample_path = options.sample or calibration.image_paths[0]
    # Produce one image with direct undistortion and one with precomputed maps.
    _, _, method_difference = undistort_sample(
        calibration,
        sample_path,
        options.output_dir,
        display=not options.no_display,
    )
    # Print enough precision to expose the version-specific interpolation change.
    print(
        "Undistortion method mean absolute difference: "
        f"{method_difference:.9f}"
    )

    # Custom exploratory runs need not meet the bundled fixture's validation.
    if options.validate:
        # Check the calibration result before trusting it for rectification.
        validate_calibration(calibration)
        # The two documented workflows should remain visually equivalent.
        if method_difference >= MAXIMUM_METHOD_DIFFERENCE:
            raise RuntimeError(
                "Direct undistortion and remapping differ by more than expected"
            )
        # Tests require an unambiguous marker after every check succeeds.
        print("Validation passed")


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and convert operational failures into a clean exit status."""

    try:
        # Let argparse keep its standard help and malformed-option behavior.
        options = parse_args(arguments)
        # Exercise the same run function used by direct imports and tests.
        run(options)
    except Exception as error:
        # imread, imwrite, OpenCV, and validation failures share one clear format.
        print(f"Error: {error}", file=sys.stderr)
        return 1
    # Zero communicates successful image generation and optional validation.
    return 0


if __name__ == "__main__":
    # SystemExit exposes main's status to shells, CTest, and subprocess tests.
    raise SystemExit(main())

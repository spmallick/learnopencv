#!/usr/bin/env python3
"""Find and annotate the centroids of multiple foreground blobs."""

from __future__ import annotations

# argparse provides a discoverable command-line interface for the example.
import argparse
# math supplies a finite-number check for the minimum contour area.
import math
# sys separates error messages from stable result output.
import sys
# Path keeps bundled inputs independent of the current working directory.
from pathlib import Path
# Optional keeps type annotations compatible with the documented Python 3.9.
from typing import Optional

# OpenCV supplies contours, geometry, image I/O, annotation, and display.
import cv2

# Reuse the exact thresholding semantics taught by the single-blob example.
try:
    # Package-style imports work when a surrounding runner imports this directory.
    from .single_blob import create_binary_mask
except ImportError:
    # Direct script execution places this directory, not its parent, on sys.path.
    from single_blob import create_binary_mask


# Resolve tracked assets relative to this file rather than the caller's cwd.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
# Use the tutorial's six-blob image when no explicit input is supplied.
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "multiple-blob.png"
# --output-dir always creates this one predictable artifact.
DEFAULT_OUTPUT_NAME = "multiple_blob_result.png"
# Sort by geometry and validate this cross-version centroid sequence.
EXPECTED_CENTROIDS = (
    (72, 130),
    (256, 115),
    (439, 115),
    (625, 115),
    (807, 122),
    (993, 116),
)
# Contour areas are stable for the tracked lossless PNG in both tested majors.
EXPECTED_AREAS = (10_194.0, 13_946.0, 14_583.0, 16_105.0, 3_688.0, 14_853.0)
# This mask statistic detects polarity or threshold changes.
EXPECTED_FOREGROUND_PIXELS = 70_791


def find_blob_centroids(binary_mask, min_area: float = 1.0):
    """Return ``(contour, centroid, area)`` entries sorted left to right."""
    # Reject an absent mask before findContours reports an implementation error.
    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("The binary mask is empty")
    # NaN and infinity would make the filtering rule misleading.
    if not math.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be a finite non-negative number")

    # Pass a copy because contour extraction has historically had differing input
    # mutation behavior, and later validation still needs the original mask.
    contours, _ = cv2.findContours(
        binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Collect only meaningful regions that satisfy the documented area filter.
    blobs = []
    # Each external contour corresponds to one candidate foreground component.
    for contour in contours:
        # contourArea is the geometric area enclosed by the contour boundary.
        area = cv2.contourArea(contour)
        # Skip noise before spending work on moments and drawing.
        if area < min_area:
            continue
        # Contour moments provide area and the first-order coordinate sums.
        moments = cv2.moments(contour, binaryImage=False)
        # A line or point contour has no enclosed area and therefore no centroid.
        if abs(moments["m00"]) <= 1e-12:
            continue
        # Normalize the first-order x moment by the contour's signed area.
        center_x = int(round(moments["m10"] / moments["m00"]))
        # Normalize the first-order y moment by the contour's signed area.
        center_y = int(round(moments["m01"] / moments["m00"]))
        # Preserve the contour for annotation and the area for reporting.
        blobs.append((contour, (center_x, center_y), area))

    # findContours does not promise outer-vector order, so impose a geometric one.
    blobs.sort(key=lambda blob: (blob[1][0], blob[1][1], blob[2]))
    # The stable list can now be compared between OpenCV versions and languages.
    return blobs


def annotate_blobs(image, blobs):
    """Return a copy of the image with contours and centroids marked."""
    # Drawing on a clone keeps the input available for independent checks.
    annotated = image.copy()
    # Human-readable labels use the same left-to-right order printed by the CLI.
    for index, (contour, centroid, _) in enumerate(blobs, start=1):
        # Outline the detected region in the tutorial's blue-green color.
        cv2.drawContours(annotated, [contour], -1, (167, 151, 0), 2)
        # Mark the exact moment centroid with a filled white dot.
        cv2.circle(annotated, centroid, 5, (255, 255, 255), cv2.FILLED)
        # Number each region so the image can be matched to terminal metrics.
        cv2.putText(
            annotated,
            str(index),
            (centroid[0] + 8, centroid[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_8,
        )
    # Return the complete annotation without forcing a GUI window.
    return annotated


def process_image(
    image,
    threshold_value: int = 127,
    foreground: str = "dark",
    min_area: float = 1.0,
):
    """Return ``(annotated_image, blobs, binary_mask)``."""
    # Use the same foreground-polarity contract as the single-blob example.
    binary_mask = create_binary_mask(image, threshold_value, foreground)
    # Measure only external blob contours, filtering any undersized regions.
    blobs = find_blob_centroids(binary_mask, min_area)
    # Expose all intermediate results for unit and cross-language validation.
    return annotate_blobs(image, blobs), blobs, binary_mask


def validate_bundled_result(input_path: Path, image, binary_mask, blobs) -> None:
    """Raise ValueError unless the bundled-image regression checks pass."""
    # Known dimensions and measurements apply only to the tracked input asset.
    if input_path.resolve() != DEFAULT_INPUT_PATH.resolve():
        raise ValueError("--validate requires the bundled multiple-blob.png input")
    # Check decoding before accepting any downstream contour measurement.
    if image.shape != (236, 1089, 3):
        raise ValueError(f"Unexpected multiple-blob.png shape: {image.shape}")
    # Compare stable, sorted geometry instead of findContours' unspecified order.
    centroids = tuple(blob[1] for blob in blobs)
    if centroids != EXPECTED_CENTROIDS:
        raise ValueError(
            f"Expected centroids {EXPECTED_CENTROIDS}, received {centroids}"
        )
    # Exact PNG geometry should produce the same areas in 4.14 and 5.0.
    areas = tuple(blob[2] for blob in blobs)
    if any(
        abs(actual - expected) > 1e-6
        for actual, expected in zip(areas, EXPECTED_AREAS)
    ) or len(areas) != len(EXPECTED_AREAS):
        raise ValueError(f"Expected areas {EXPECTED_AREAS}, received {areas}")
    # Verify the thresholded foreground as well as its extracted boundaries.
    foreground_pixels = cv2.countNonZero(binary_mask)
    if foreground_pixels != EXPECTED_FOREGROUND_PIXELS:
        raise ValueError(
            "Expected "
            f"{EXPECTED_FOREGROUND_PIXELS} foreground pixels, "
            f"received {foreground_pixels}"
        )


def resolve_output_path(
    output: Optional[Path], output_directory: Optional[Path]
) -> Optional[Path]:
    """Create the requested directory and return the final output path."""
    # Preserve the explicit filename accepted by the existing migration draft.
    if output is not None:
        # Nested output paths should work without a manual mkdir step.
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    # Omitting both output switches leaves the filesystem untouched.
    if output_directory is None:
        return None
    # A deterministic directory artifact makes exact output-set tests possible.
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / DEFAULT_OUTPUT_NAME


def write_annotated_image(output_path: Path, annotated, validate: bool) -> None:
    """Write an annotation and optionally prove OpenCV can read it back."""
    # imwrite can either return False or raise cv2.error depending on the failure.
    try:
        written = cv2.imwrite(str(output_path), annotated)
    except cv2.error as error:
        raise OSError(f"could not write output image '{output_path}': {error}") from error
    # Turn a silent encoder failure into a useful nonzero CLI exit.
    if not written:
        raise OSError(f"could not write output image '{output_path}'")
    # Regression mode verifies that the actual encoded artifact is readable.
    if validate:
        decoded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != annotated.shape:
            raise OSError(f"could not verify output image '{output_path}'")


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command-line interface in one testable function."""
    # argparse supplies help output and consistent invalid-argument exits.
    parser = argparse.ArgumentParser(description="Find centers of multiple blobs.")
    # --ipimage preserves old article commands; --input is the preferred spelling.
    parser.add_argument(
        "-i",
        "--input",
        "--ipimage",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="input image (default: bundled multiple-blob.png)",
    )
    # The default threshold separates the dark tutorial shapes from white.
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="grayscale threshold from 0 to 255 (default: 127)",
    )
    # Supporting both polarities makes the moment semantics explicit.
    parser.add_argument(
        "--foreground",
        choices=("dark", "light"),
        default="dark",
        help="whether blobs are darker or lighter than the background",
    )
    # Filter small external contours without changing the source image.
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="ignore contours smaller than this finite area (default: 1)",
    )
    # Only one output naming mode may be selected for an invocation.
    output_group = parser.add_mutually_exclusive_group()
    # Keep the explicit output path implemented by the migration draft.
    output_group.add_argument(
        "--output",
        type=Path,
        help="optional path for the annotated image",
    )
    # Add the standardized directory-based output used by tests and releases.
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help=f"write {DEFAULT_OUTPUT_NAME} in this directory",
    )
    # A headless switch is required for servers and automated test runners.
    parser.add_argument(
        "--no-display", action="store_true", help="do not open a GUI window"
    )
    # Validation checks the tracked asset's complete metric set.
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate the bundled image and print a success marker",
    )
    # Return the parser so tests can inspect parsing without running OpenCV.
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute one CLI invocation and return a process-style status code."""
    # Load a color image because contours will be drawn in color.
    image = cv2.imread(str(args.input_path), cv2.IMREAD_COLOR)
    # Stop with a clear error if the path is missing or the codec cannot read it.
    if image is None:
        print(
            f"Error: could not read input image '{args.input_path}'",
            file=sys.stderr,
        )
        return 1

    try:
        # Run the same implementation that direct unit tests exercise.
        annotated, blobs, binary_mask = process_image(
            image, args.threshold, args.foreground, args.min_area
        )
        # An empty filtered set is an input/settings error, not successful output.
        if not blobs:
            raise ValueError("no blobs matched the threshold and area settings")
        # Determine and create the optional destination before reporting success.
        output_path = resolve_output_path(args.output, args.output_dir)
        # Regression mode verifies every stable metric of the bundled sample.
        if args.validate:
            validate_bundled_result(args.input_path, image, binary_mask, blobs)
        # Save and, in validation mode, decode the generated artifact.
        if output_path is not None:
            write_annotated_image(output_path, annotated, args.validate)
    # ValueError covers settings/metrics and OSError covers filesystem/encoding.
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Print all blobs in the same deterministic order used for annotation.
    for index, (_, centroid, area) in enumerate(blobs, start=1):
        print(
            f"Blob {index}: centroid=({centroid[0]}, {centroid[1]}), "
            f"area={area:.1f}"
        )
    # Version evidence prevents an accidental 4.x/5.x environment mix-up.
    print(f"OpenCV version: {cv2.__version__}")
    # Tests require an explicit marker rather than relying on exit code alone.
    if args.validate:
        print("VALIDATION PASSED: center_of_multiple_blob")

    # Interactive display remains the tutorial default but is disabled by tests.
    if not args.no_display:
        cv2.imshow("Blob centroids", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Successful processing and optional validation use the conventional zero exit.
    return 0


def main() -> int:
    """Parse arguments and run the example."""
    # Importing this module remains side-effect free because parsing happens here.
    parser = build_argument_parser()
    # Convert raw process arguments into documented types and choices.
    args = parser.parse_args()
    # argparse converts the integer but does not enforce its numeric range.
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    # Reject negative, NaN, or infinite filtering thresholds before image I/O.
    if not math.isfinite(args.min_area) or args.min_area < 0:
        parser.error("--min-area must be a finite non-negative number")
    # Delegate execution so tests can construct a namespace directly if needed.
    return run(args)


# Execute only when launched as a script, never merely when imported by tests.
if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Find and annotate the centroid of one foreground blob."""

from __future__ import annotations

# argparse provides a documented command-line interface for the tutorial.
import argparse
# sys lets errors use stderr while normal results remain on stdout.
import sys
# Path makes bundled assets independent of the caller's working directory.
from pathlib import Path
# Optional keeps type annotations compatible with the documented Python 3.9.
from typing import Optional

# OpenCV supplies image I/O, thresholding, moments, drawing, and display APIs.
import cv2


# Resolve the example directory once so the default image works from any cwd.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
# Use the tutorial's tracked circle image when the caller omits --input.
DEFAULT_INPUT_PATH = SCRIPT_DIRECTORY / "circle.png"
# Give --output-dir a deterministic filename that tests can assert exactly.
DEFAULT_OUTPUT_NAME = "single_blob_result.png"
# The tracked image has a stable foreground centroid under the documented mask.
EXPECTED_CENTROID = (112, 112)
# This count catches accidental foreground/background inversions.
EXPECTED_FOREGROUND_PIXELS = 35_272


def create_binary_mask(
    image, threshold_value: int = 127, foreground: str = "dark"
):
    """Convert a BGR image to a mask whose foreground pixels are white."""
    # Reject an absent or zero-sized image before cvtColor emits a cryptic error.
    if image is None or image.size == 0:
        raise ValueError("The input image is empty")
    # OpenCV thresholds use the inclusive 8-bit intensity range.
    if not 0 <= threshold_value <= 255:
        raise ValueError("threshold_value must be between 0 and 255")
    # Keep the polarity choice explicit instead of silently guessing.
    if foreground not in {"dark", "light"}:
        raise ValueError("foreground must be 'dark' or 'light'")

    # Moments need a one-channel mask, so discard color after loading the image.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Dark objects must be inverted so the intended blob, not its background,
    # becomes the nonzero foreground measured by cv2.moments.
    threshold_type = (
        cv2.THRESH_BINARY_INV if foreground == "dark" else cv2.THRESH_BINARY
    )
    # Convert the grayscale image into an 8-bit mask containing only 0 and 255.
    _, binary_mask = cv2.threshold(
        gray_image, threshold_value, 255, threshold_type
    )
    # Return the mask as well as using it internally so tests can inspect it.
    return binary_mask


def find_centroid(binary_mask) -> tuple[int, int]:
    """Return the centroid of the nonzero pixels in a binary mask."""
    # binaryImage=True treats every nonzero pixel equally, independent of value.
    moments = cv2.moments(binary_mask, binaryImage=True)
    # m00 is the foreground area; division by zero means no blob was detected.
    if abs(moments["m00"]) <= 1e-12:
        raise ValueError("No foreground pixels were found in the binary mask")

    # Normalize the first-order x moment by area to obtain the horizontal center.
    center_x = int(round(moments["m10"] / moments["m00"]))
    # Normalize the first-order y moment by area to obtain the vertical center.
    center_y = int(round(moments["m01"] / moments["m00"]))
    # A small immutable tuple is convenient for both drawing and validation.
    return center_x, center_y


def annotate_centroid(image, centroid: tuple[int, int]):
    """Return a copy of the image with its centroid marked."""
    # Never paint over the caller's input; preserving it simplifies comparisons.
    annotated = image.copy()
    # Unpack the coordinates for the readable text-offset calculation below.
    center_x, center_y = centroid
    # Draw a filled white dot directly over the measured center.
    cv2.circle(annotated, centroid, 5, (255, 255, 255), cv2.FILLED)
    # Label the point using constants available in both OpenCV 4.14 and 5.
    cv2.putText(
        annotated,
        "centroid",
        (center_x - 25, center_y - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_8,
    )
    # Return the annotated copy so GUI display remains optional.
    return annotated


def process_image(
    image, threshold_value: int = 127, foreground: str = "dark"
):
    """Return ``(annotated_image, centroid, binary_mask)``."""
    # Build a foreground mask with the selected threshold and polarity.
    binary_mask = create_binary_mask(image, threshold_value, foreground)
    # Compute the center from mask pixels rather than from the white background.
    centroid = find_centroid(binary_mask)
    # Keep all useful intermediate results available to tests and other callers.
    return annotate_centroid(image, centroid), centroid, binary_mask


def validate_bundled_result(
    input_path: Path, image, binary_mask, centroid: tuple[int, int]
) -> None:
    """Raise ValueError unless the bundled-image regression checks pass."""
    # Validation values apply only to the immutable tutorial asset.
    if input_path.resolve() != DEFAULT_INPUT_PATH.resolve():
        raise ValueError("--validate requires the bundled circle.png input")
    # Check the decoded dimensions before trusting any derived measurement.
    if image.shape != (225, 225, 3):
        raise ValueError(f"Unexpected circle.png shape: {image.shape}")
    # A centroid mismatch would change the tutorial's central numerical result.
    if centroid != EXPECTED_CENTROID:
        raise ValueError(
            f"Expected centroid {EXPECTED_CENTROID}, received {centroid}"
        )
    # Pixel count also detects a threshold-polarity regression that the centered
    # source image could otherwise hide because its background has the same center.
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
    # An explicit filename is useful for backward-compatible one-off commands.
    if output is not None:
        # Create its parent as well, so nested paths behave predictably.
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    # With no output option, the example only prints and optionally displays.
    if output_directory is None:
        return None
    # CI uses a fresh directory and expects one exact, deterministic filename.
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory / DEFAULT_OUTPUT_NAME


def write_annotated_image(output_path: Path, annotated, validate: bool) -> None:
    """Write an annotation and optionally prove OpenCV can read it back."""
    # imwrite returns False for many failures instead of raising an exception.
    try:
        written = cv2.imwrite(str(output_path), annotated)
    except cv2.error as error:
        raise OSError(f"could not write output image '{output_path}': {error}") from error
    # Convert the boolean failure into a clear tutorial-facing message.
    if not written:
        raise OSError(f"could not write output image '{output_path}'")
    # Validation checks the actual encoded file, not just imwrite's return value.
    if validate:
        decoded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != annotated.shape:
            raise OSError(f"could not verify output image '{output_path}'")


def build_argument_parser() -> argparse.ArgumentParser:
    """Describe the command-line interface in one testable function."""
    # Defaults in ArgumentParser automatically appear in --help output.
    parser = argparse.ArgumentParser(description="Find the center of a single blob.")
    # --ipimage keeps old article commands working; --input is the preferred name.
    parser.add_argument(
        "-i",
        "--input",
        "--ipimage",
        dest="input_path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="input image (default: bundled circle.png)",
    )
    # Expose the tutorial threshold so other high-contrast images can be tested.
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="grayscale threshold from 0 to 255 (default: 127)",
    )
    # The default matches the bundled dark circle on a light background.
    parser.add_argument(
        "--foreground",
        choices=("dark", "light"),
        default="dark",
        help="whether the blob is darker or lighter than the background",
    )
    # Only one output naming mode may be selected for an invocation.
    output_group = parser.add_mutually_exclusive_group()
    # Preserve the draft migration's explicit output-file interface.
    output_group.add_argument(
        "--output",
        type=Path,
        help="optional path for the annotated image",
    )
    # Add the standard migration interface used by repeatable test runs.
    output_group.add_argument(
        "--output-dir",
        type=Path,
        help=f"write {DEFAULT_OUTPUT_NAME} in this directory",
    )
    # Headless operation prevents imshow/waitKey from blocking automated tests.
    parser.add_argument(
        "--no-display", action="store_true", help="do not open a GUI window"
    )
    # Validation checks the known metrics of the bundled image.
    parser.add_argument(
        "--validate",
        action="store_true",
        help="validate the bundled image and print a success marker",
    )
    # Return the parser so tests can inspect parsing without running OpenCV.
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute one CLI invocation and return a process-style status code."""
    # Decode in color because the annotation is drawn on a three-channel image.
    image = cv2.imread(str(args.input_path), cv2.IMREAD_COLOR)
    # A missing or unsupported image should fail before any OpenCV operation.
    if image is None:
        print(
            f"Error: could not read input image '{args.input_path}'",
            file=sys.stderr,
        )
        return 1

    try:
        # Exercise the same callable path used by unit tests.
        annotated, centroid, binary_mask = process_image(
            image, args.threshold, args.foreground
        )
        # Resolve and create output paths before reporting successful completion.
        output_path = resolve_output_path(args.output, args.output_dir)
        # Validate stable tutorial metrics when the caller requests regression mode.
        if args.validate:
            validate_bundled_result(
                args.input_path, image, binary_mask, centroid
            )
        # Encode the annotation only when an output option was supplied.
        if output_path is not None:
            write_annotated_image(output_path, annotated, args.validate)
    # ValueError covers validation/input semantics; OSError covers output paths.
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Print the primary result in a stable form shared with the C++ example.
    print(f"Centroid: ({centroid[0]}, {centroid[1]})")
    # Emit the exact runtime version to make matrix evidence unambiguous.
    print(f"OpenCV version: {cv2.__version__}")
    # A machine-readable marker lets CTest-style harnesses reject silent success.
    if args.validate:
        print("VALIDATION PASSED: single_blob")

    # Keep visualization opt-in for CI but on by default for tutorial use.
    if not args.no_display:
        cv2.imshow("Image with center", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Zero communicates that processing, optional output, and validation succeeded.
    return 0


def main() -> int:
    """Parse arguments and run the example."""
    # Construct the parser inside main so importing the module has no side effects.
    parser = build_argument_parser()
    # Parse the process command line using argparse's standard error reporting.
    args = parser.parse_args()
    # Keep threshold validation explicit because argparse only converts the type.
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    # Hand the validated namespace to a directly testable execution function.
    return run(args)


# Importing this file exposes its functions without opening windows or reading data.
if __name__ == "__main__":
    raise SystemExit(main())

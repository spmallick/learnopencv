"""Detect and decode one QR code with OpenCV 4.14 or OpenCV 5."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "qrcode-learnopencv.jpg"
EXPECTED_DATA = "http://LearnOpenCV.com"


@dataclass(frozen=True)
class DetectionResult:
    """The decoded text, four corners, straightened code, and elapsed time."""

    data: str
    corners: np.ndarray | None
    rectified: np.ndarray | None
    elapsed_seconds: float


def read_image(path: Path) -> np.ndarray:
    """Read an image and fail with a useful path when decoding is impossible."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {path}")
    return image


def detect_qr_code(image: np.ndarray) -> DetectionResult:
    """Run OpenCV's single-code detector and normalize its corner layout."""

    detector = cv2.QRCodeDetector()
    started = time.perf_counter()
    data, points, straight_code = detector.detectAndDecode(image)
    elapsed_seconds = time.perf_counter() - started

    corners = None
    if points is not None:
        corners = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    rectified = None
    if straight_code is not None and straight_code.size:
        rectified = np.asarray(straight_code, dtype=np.uint8)

    return DetectionResult(data, corners, rectified, elapsed_seconds)


def draw_corners(image: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    """Return a copy with the detected quadrilateral drawn in blue."""

    annotated = image.copy()
    if corners is None:
        return annotated

    polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [polygon], True, (255, 0, 0), 3, cv2.LINE_AA)
    return annotated


def validate_result(result: DetectionResult) -> None:
    """Check stable facts supplied by the bundled LearnOpenCV QR image."""

    if result.data != EXPECTED_DATA:
        raise RuntimeError(
            f"Decoded payload mismatch: expected {EXPECTED_DATA!r}, "
            f"got {result.data!r}"
        )
    if result.corners is None or result.corners.shape != (4, 2):
        raise RuntimeError(
            "Expected exactly four QR-code corners, got "
            f"{None if result.corners is None else result.corners.shape}"
        )
    if result.rectified is None or result.rectified.size == 0:
        raise RuntimeError("OpenCV returned no rectified QR-code image")


def run(
    input_path: Path,
    output_dir: Path,
    *,
    show_windows: bool,
    validate: bool,
) -> DetectionResult:
    """Decode an image, save reproducible outputs, and optionally display them."""

    image = read_image(input_path)
    result = detect_qr_code(image)
    annotated = draw_corners(image, result.corners)

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "qr-code-annotated.png"
    if not cv2.imwrite(str(annotated_path), annotated):
        raise OSError(f"Unable to write output image: {annotated_path}")

    if result.rectified is not None:
        rectified_path = output_dir / "qr-code-rectified.png"
        if not cv2.imwrite(str(rectified_path), result.rectified):
            raise OSError(f"Unable to write rectified image: {rectified_path}")

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Detect and decode time: {result.elapsed_seconds:.6f} seconds")
    print(f"Decoded data: {result.data or '<none>'}")
    print(f"Annotated image: {annotated_path}")

    if validate:
        validate_result(result)
        print("VALIDATION PASSED: payload and four QR corners match")

    if show_windows:
        cv2.imshow("QR code result", annotated)
        if result.rectified is not None:
            cv2.imshow("Rectified QR code", result.rectified)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line controls shared by normal and automated runs."""

    parser = argparse.ArgumentParser(
        description="Detect and decode a QR code with OpenCV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image (default: {DEFAULT_INPUT.name}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for annotated and rectified PNG files.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open GUI windows; required on headless systems.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check the expected payload and corner count for the bundled image.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line application and report user-facing failures."""

    args = parse_args()
    try:
        run(
            args.input.resolve(),
            args.output_dir.resolve(),
            show_windows=not args.no_display,
            validate=args.validate,
        )
    except (FileNotFoundError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

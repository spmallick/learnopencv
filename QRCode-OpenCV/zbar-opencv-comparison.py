"""Compare OpenCV and optional ZBar QR decoding on one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "qrcode-learnopencv.jpg"
EXPECTED_DATA = "http://LearnOpenCV.com"


def load_pyzbar():
    """Import the optional ZBar binding only when this comparison is run."""

    try:
        import pyzbar.pyzbar as pyzbar
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "ZBar comparison requires pyzbar and the system zbar library. "
            "See README.md for installation commands."
        ) from error
    return pyzbar


def polygon_points(decoded_object) -> np.ndarray:
    """Normalize a ZBar polygon to integer OpenCV drawing coordinates."""

    points = np.array(
        [(point.x, point.y) for point in decoded_object.polygon],
        dtype=np.float32,
    )
    if len(points) > 4:
        points = cv2.convexHull(points).reshape(-1, 2)
    return np.rint(points).astype(np.int32)


def run(
    input_path: Path,
    output_path: Path,
    *,
    show_window: bool,
    validate: bool,
) -> tuple[str, str]:
    """Decode with both libraries, annotate the image, and return both texts."""

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {input_path}")

    pyzbar = load_pyzbar()
    zbar_objects = pyzbar.decode(image)
    zbar_text = (
        zbar_objects[0].data.decode("utf-8", errors="replace")
        if zbar_objects
        else ""
    )

    opencv_text, opencv_points, _ = cv2.QRCodeDetector().detectAndDecode(image)
    annotated = image.copy()

    for decoded_object in zbar_objects:
        cv2.polylines(
            annotated,
            [polygon_points(decoded_object).reshape(-1, 1, 2)],
            True,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if opencv_points is not None:
        opencv_polygon = np.rint(opencv_points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            annotated,
            [opencv_polygon],
            True,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"ZBar: {zbar_text or 'not detected'}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"OpenCV: {opencv_text or 'not detected'}",
        (10, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 180, 0),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Unable to write comparison image: {output_path}")

    print(f"OpenCV version: {cv2.__version__}")
    print(f"ZBar decoded data: {zbar_text or '<none>'}")
    print(f"OpenCV decoded data: {opencv_text or '<none>'}")
    print(f"Comparison image: {output_path}")

    if validate:
        if zbar_text != EXPECTED_DATA or opencv_text != EXPECTED_DATA:
            raise RuntimeError(
                "The bundled image must decode to the expected payload in "
                "both ZBar and OpenCV"
            )
        print("VALIDATION PASSED: OpenCV and ZBar payloads match")

    if show_window:
        cv2.imshow("OpenCV and ZBar comparison", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return zbar_text, opencv_text


def parse_args() -> argparse.Namespace:
    """Parse deterministic image-comparison command-line options."""

    parser = argparse.ArgumentParser(
        description="Compare OpenCV QRCodeDetector with the optional ZBar decoder."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/zbar-opencv-comparison.png"),
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the comparison and convert expected failures to a nonzero exit."""

    args = parse_args()
    try:
        run(
            args.input.resolve(),
            args.output.resolve(),
            show_window=not args.no_display,
            validate=args.validate,
        )
    except (FileNotFoundError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

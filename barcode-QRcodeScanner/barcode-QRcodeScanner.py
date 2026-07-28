#!/usr/bin/env python3
"""Decode QR codes and supported retail barcodes with OpenCV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


@dataclass(frozen=True)
class CodeResult:
    code_type: str
    data: str
    points: NDArray[np.float32]


def _normalise_polygons(points: object) -> list[NDArray[np.float32]]:
    if points is None:
        return []
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return []
    return [polygon.reshape(-1, 2) for polygon in array.reshape(-1, 4, 2)]


def decode_qr_codes(image: Image) -> list[CodeResult]:
    """Decode every QR code OpenCV can resolve in an image."""
    if image is None or image.size == 0:
        raise ValueError("decode_qr_codes expects a non-empty image")

    detector = cv2.QRCodeDetector()
    detected, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
    polygons = _normalise_polygons(points)
    results = [
        CodeResult("QR_CODE", data, polygon)
        for data, polygon in zip(decoded_info if detected else (), polygons)
        if data
    ]
    if results:
        return results

    # Some OpenCV builds detect a single code more reliably through the
    # single-code path, so retain this deterministic fallback.
    data, single_points, _ = detector.detectAndDecode(image)
    single_polygons = _normalise_polygons(single_points)
    if data and single_polygons:
        return [CodeResult("QR_CODE", data, single_polygons[0])]
    return []


def _create_barcode_detector() -> object:
    namespace = getattr(cv2, "barcode", None)
    if namespace is not None and hasattr(namespace, "BarcodeDetector"):
        return namespace.BarcodeDetector()
    legacy_alias = getattr(cv2, "barcode_BarcodeDetector", None)
    if legacy_alias is not None:
        return legacy_alias()
    raise RuntimeError(
        "This OpenCV build has no BarcodeDetector. Install opencv-python "
        "4.8+ or an opencv-contrib-python build that includes barcode."
    )


def decode_barcodes(image: Image) -> list[CodeResult]:
    """Decode EAN-8, EAN-13, UPC-A, and UPC-E barcodes supported by OpenCV."""
    if image is None or image.size == 0:
        raise ValueError("decode_barcodes expects a non-empty image")

    detector = _create_barcode_detector()
    detected, decoded_info, decoded_types, points = (
        detector.detectAndDecodeWithType(image)
    )
    polygons = _normalise_polygons(points)
    if not detected:
        return []
    return [
        CodeResult(code_type, data, polygon)
        for data, code_type, polygon in zip(
            decoded_info, decoded_types, polygons
        )
        if data
    ]


def annotate_codes(image: Image, results: Iterable[CodeResult]) -> Image:
    """Draw decoded quadrangles and readable labels on a copy of the image."""
    annotated = image.copy()
    for result in results:
        polygon = np.rint(result.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [polygon], True, (30, 190, 30), 3, cv2.LINE_AA)
        anchor_x = int(np.min(polygon[:, 0, 0]))
        anchor_y = max(24, int(np.min(polygon[:, 0, 1])) - 10)
        label = f"{result.code_type}: {result.data}"
        cv2.putText(
            annotated,
            label,
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (25, 25, 220),
            2,
            cv2.LINE_AA,
        )
    return annotated


def scan_image(
    image: Image, *, scan_qr: bool = True, scan_barcodes: bool = True
) -> list[CodeResult]:
    results: list[CodeResult] = []
    if scan_qr:
        results.extend(decode_qr_codes(image))
    if scan_barcodes:
        results.extend(decode_barcodes(image))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode QR codes plus EAN/UPC retail barcodes using native "
            "OpenCV detectors."
        )
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("zbar-test.jpg"),
        help="Input image (default: bundled zbar-test.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/decoded-codes.png"),
        help="Annotated output image",
    )
    parser.add_argument("--no-qr", action="store_true", help="Skip QR decoding")
    parser.add_argument(
        "--no-barcode", action="store_true", help="Skip EAN/UPC barcode decoding"
    )
    parser.add_argument("--display", action="store_true", help="Open a GUI window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.no_qr and args.no_barcode:
        print("error: both decoders are disabled")
        return 2

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not read input image: {args.image}")
        return 2

    try:
        results = scan_image(
            image, scan_qr=not args.no_qr, scan_barcodes=not args.no_barcode
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    for result in results:
        print(f"{result.code_type}\t{result.data}")
    print(f"decoded_count={len(results)}")

    try:
        annotated = annotate_codes(image, results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), annotated):
            raise OSError(f"OpenCV could not write output image: {args.output}")
    except (OSError, cv2.error) as error:
        print(f"error: {error}")
        return 2
    print(f"output={args.output.resolve()}")

    if args.display:
        cv2.imshow("Decoded codes", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

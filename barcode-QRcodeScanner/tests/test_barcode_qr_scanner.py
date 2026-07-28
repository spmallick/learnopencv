from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "barcode-QRcodeScanner.py"
SPEC = importlib.util.spec_from_file_location("barcode_qr_scanner", MODULE_PATH)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


LEFT_ODD = (
    "0001101",
    "0011001",
    "0010011",
    "0111101",
    "0100011",
    "0110001",
    "0101111",
    "0111011",
    "0110111",
    "0001011",
)
LEFT_EVEN = (
    "0100111",
    "0110011",
    "0011011",
    "0100001",
    "0011101",
    "0111001",
    "0000101",
    "0010001",
    "0001001",
    "0010111",
)
RIGHT = tuple(
    "".join("1" if bit == "0" else "0" for bit in pattern)
    for pattern in LEFT_ODD
)
PARITY = (
    "LLLLLL",
    "LLGLGG",
    "LLGGLG",
    "LLGGGL",
    "LGLLGG",
    "LGGLLG",
    "LGGGLL",
    "LGLGLG",
    "LGLGGL",
    "LGGLGL",
)


def make_ean13(code: str = "5901234123457") -> np.ndarray:
    if len(code) != 13 or not code.isdigit():
        raise ValueError("EAN-13 test data must contain exactly 13 digits")
    expected_check = (
        10
        - (
            sum(int(digit) for digit in code[:12:2])
            + 3 * sum(int(digit) for digit in code[1:12:2])
        )
        % 10
    ) % 10
    if expected_check != int(code[-1]):
        raise ValueError("EAN-13 test data has an invalid check digit")

    left = "".join(
        (LEFT_ODD if encoding == "L" else LEFT_EVEN)[int(digit)]
        for encoding, digit in zip(PARITY[int(code[0])], code[1:7])
    )
    right = "".join(RIGHT[int(digit)] for digit in code[7:])
    bits = "101" + left + "01010" + right + "101"

    module_width = 2
    quiet_modules = 20
    gray = np.full(
        (180, (len(bits) + 2 * quiet_modules) * module_width),
        255,
        dtype=np.uint8,
    )
    for index, bit in enumerate(bits):
        if bit == "1":
            x0 = (quiet_modules + index) * module_width
            gray[20:120, x0 : x0 + module_width] = 0
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class BarcodeQrScannerTests(unittest.TestCase):
    def test_bundled_fixture_decodes_qr(self) -> None:
        image = cv2.imread(str(PROJECT / "zbar-test.jpg"), cv2.IMREAD_COLOR)
        results = scanner.decode_qr_codes(image)
        self.assertEqual([(item.code_type, item.data) for item in results], [
            ("QR_CODE", "http://LearnOpenCV.com")
        ])
        self.assertEqual(results[0].points.shape, (4, 2))

    def test_synthetic_ean13_decodes_with_native_detector(self) -> None:
        results = scanner.decode_barcodes(make_ean13())
        self.assertEqual(
            [(item.code_type, item.data) for item in results],
            [("EAN_13", "5901234123457")],
        )

    def test_annotation_preserves_shape(self) -> None:
        image = cv2.imread(str(PROJECT / "zbar-test.jpg"), cv2.IMREAD_COLOR)
        results = scanner.decode_qr_codes(image)
        annotated = scanner.annotate_codes(image, results)
        self.assertEqual(annotated.shape, image.shape)
        self.assertFalse(np.array_equal(annotated, image))


if __name__ == "__main__":
    unittest.main()

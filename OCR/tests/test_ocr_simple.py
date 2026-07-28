from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import ocr_simple  # noqa: E402


class OcrSimpleTests(unittest.TestCase):
    def test_preprocessing_modes_preserve_geometry(self) -> None:
        image = np.full((120, 420, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "OCR",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        for mode in ocr_simple.PREPROCESSING_MODES:
            with self.subTest(mode=mode):
                prepared = ocr_simple.preprocess_image(image, mode)
                self.assertEqual(prepared.shape[:2], image.shape[:2])
        otsu = ocr_simple.preprocess_image(image, "otsu")
        self.assertTrue(set(np.unique(otsu)).issubset({0, 255}))

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract is not installed")
    def test_tesseract_recognizes_synthetic_line(self) -> None:
        image = np.full((180, 900, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "OPENCV 2026",
            (35, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        prepared = ocr_simple.preprocess_image(image, "gray")
        text = ocr_simple.run_tesseract(prepared, oem=1, psm=7)
        self.assertIn("OPENCV 2026", " ".join(text.split()))

    @unittest.skipUnless(shutil.which("tesseract"), "Tesseract is not installed")
    def test_bundled_sign_regression(self) -> None:
        image = cv2.imread(
            str(PROJECT / "images" / "road-sign-3.jpg"), cv2.IMREAD_COLOR
        )
        prepared = ocr_simple.preprocess_image(image, "gray")
        text = ocr_simple.run_tesseract(prepared, oem=1, psm=6)
        normalized = " ".join(text.split()).upper()
        self.assertIn("THIS PROPERTY", normalized)
        self.assertIn("VIDEO SURVEILLANCE", normalized)

    def test_rejects_unknown_preprocessing_mode(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "unknown preprocessing"):
            ocr_simple.preprocess_image(image, "magic")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import imfill  # noqa: E402


class FillHolesTests(unittest.TestCase):
    def test_fills_only_enclosed_background(self) -> None:
        binary = np.zeros((25, 25), dtype=np.uint8)
        cv2.rectangle(binary, (3, 3), (21, 21), 255, cv2.FILLED)
        cv2.circle(binary, (12, 12), 5, 0, cv2.FILLED)

        expected = np.zeros_like(binary)
        cv2.rectangle(expected, (3, 3), (21, 21), 255, cv2.FILLED)
        filled, flooded, holes = imfill.fill_holes(binary)

        np.testing.assert_array_equal(filled, expected)
        self.assertEqual(cv2.countNonZero(holes), 81)
        self.assertEqual(flooded[0, 0], 255)
        self.assertEqual(flooded[12, 12], 0)

    def test_border_touching_foreground_does_not_hide_exterior(self) -> None:
        binary = np.zeros((20, 20), dtype=np.uint8)
        cv2.rectangle(binary, (0, 2), (15, 17), 255, cv2.FILLED)
        cv2.rectangle(binary, (4, 6), (10, 12), 0, cv2.FILLED)

        filled, _, holes = imfill.fill_holes(binary)
        self.assertEqual(cv2.countNonZero(holes), 49)
        self.assertEqual(filled[9, 7], 255)
        self.assertEqual(filled[0, 19], 0)

    def test_rejects_non_binary_input(self) -> None:
        non_binary = np.array([[0, 127, 255]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "binary mask"):
            imfill.fill_holes(non_binary)
        with self.assertRaisesRegex(ValueError, "non-empty uint8"):
            imfill.fill_holes(np.empty((0, 0), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "non-empty uint8"):
            imfill.threshold_foreground(
                np.zeros((3, 3), dtype=np.float32)
            )

    def test_bundled_image_regression(self) -> None:
        binary, flooded, holes, filled = imfill.run_pipeline(
            PROJECT / "nickel.jpg", 220
        )
        self.assertEqual(binary.shape, (295, 300))
        self.assertEqual(flooded.shape, binary.shape)
        self.assertGreater(cv2.countNonZero(holes), 5_000)
        self.assertEqual(
            cv2.countNonZero(filled),
            cv2.countNonZero(binary) + cv2.countNonZero(holes),
        )


if __name__ == "__main__":
    unittest.main()

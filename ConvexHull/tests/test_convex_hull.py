from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import example  # noqa: E402


class ConvexHullTests(unittest.TestCase):
    def test_concave_shape_has_larger_hull(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        points = np.array(
            [[20, 20], [140, 20], [140, 140], [90, 140], [90, 70],
             [70, 70], [70, 140], [20, 140]],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], (255, 255, 255))

        drawing, contours, hulls = example.build_convex_hull_visualization(
            image, threshold_value=127
        )

        self.assertEqual(drawing.shape, image.shape)
        self.assertGreaterEqual(len(contours), 1)
        largest = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        self.assertGreater(
            cv2.contourArea(hulls[largest]),
            cv2.contourArea(contours[largest]),
        )
        self.assertGreater(np.count_nonzero(drawing), 0)

    def test_rejects_invalid_threshold(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "between 0 and 255"):
            example.build_convex_hull_visualization(image, 256)

    def test_cli_writes_readable_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "hulls.png"
            exit_code = example.main(
                [
                    "--input",
                    str(PROJECT_DIR / "sample.jpg"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            result = cv2.imread(str(output), cv2.IMREAD_COLOR)
            source = cv2.imread(str(PROJECT_DIR / "sample.jpg"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(result)
            self.assertIsNotNone(source)
            self.assertEqual(result.shape, source.shape)
            _, contours, _ = example.build_convex_hull_visualization(source)
            self.assertGreater(len(contours), 100)


if __name__ == "__main__":
    unittest.main()

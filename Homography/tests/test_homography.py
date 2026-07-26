from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import utils  # noqa: E402


BOOK_POINTS = "318,256;534,372;316,670;73,473"
BILLBOARD_POINTS = "80,165;320,255;255,390;35,305"


class HomographyUtilityTests(unittest.TestCase):
    def test_homography_maps_all_four_corners(self) -> None:
        source = utils.rectangle_corners(120, 80)
        destination = utils.parse_points("20,30;180,20;190,140;10,150")

        homography = utils.compute_homography(source, destination)
        transformed = cv2.perspectiveTransform(
            source.reshape(1, -1, 2), homography
        ).reshape(-1, 2)

        np.testing.assert_allclose(transformed, destination, atol=1e-3)

    def test_rectification_has_requested_dimensions(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[20:80, 20:100] = (0, 255, 0)
        points = utils.parse_points("20,20;99,20;99,79;20,79")

        rectified, _ = utils.rectify_image(image, points, 80, 60)

        self.assertEqual(rectified.shape, (60, 80, 3))
        self.assertGreater(int(rectified[30, 40, 1]), 240)

    def test_composite_changes_only_destination_quad(self) -> None:
        source = np.full((40, 60, 3), (0, 0, 255), dtype=np.uint8)
        destination = np.full((120, 180, 3), (255, 0, 0), dtype=np.uint8)
        points = utils.parse_points("40,30;139,30;139,89;40,89")

        result, _ = utils.composite_on_quad(
            source, destination, points
        )

        np.testing.assert_array_equal(result[5, 5], destination[5, 5])
        self.assertGreater(int(result[60, 90, 2]), 240)
        self.assertLess(int(result[60, 90, 0]), 10)

    def test_invalid_point_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero area|convex"):
            utils.parse_points("0,0;10,10;0,10;10,0")


class HomographyCliTests(unittest.TestCase):
    def run_script(
        self,
        script_name: str,
        output: Path,
        *arguments: str,
    ) -> np.ndarray:
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / script_name),
                *arguments,
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Saved", completed.stdout)
        image = cv2.imread(str(output), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        return image

    def test_all_python_examples_run_headlessly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            perspective = self.run_script(
                "perspective-correction.py",
                temporary_path / "perspective.jpg",
                "--points",
                BOOK_POINTS,
            )
            compact = self.run_script(
                "homography.py",
                temporary_path / "homography.jpg",
                "--points",
                BOOK_POINTS,
            )
            book = self.run_script(
                "homography_book.py",
                temporary_path / "book.jpg",
            )
            billboard = self.run_script(
                "virtual-billboard.py",
                temporary_path / "billboard.jpg",
                "--points",
                BILLBOARD_POINTS,
            )
            generic = self.run_script(
                "homography2.py",
                temporary_path / "generic.jpg",
                "--points",
                BILLBOARD_POINTS,
            )

            self.assertEqual(perspective.shape, (400, 300, 3))
            self.assertEqual(compact.shape, (400, 300, 3))
            self.assertEqual(book.shape, (800, 600, 3))
            self.assertEqual(billboard.shape, (854, 1280, 3))
            self.assertEqual(generic.shape, (854, 1280, 3))


if __name__ == "__main__":
    unittest.main()

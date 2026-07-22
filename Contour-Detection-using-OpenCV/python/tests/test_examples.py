"""Headless regression tests for the contour-detection tutorial scripts."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2


PYTHON_DIR = Path(__file__).resolve().parents[1]


class ContourExamplesTest(unittest.TestCase):
    def run_example(self, relative_script: str, expected_outputs: set[str]) -> str:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            command = [
                sys.executable,
                str(PYTHON_DIR / relative_script),
                "--no-display",
                "--validate",
                "--output-dir",
                str(output_dir),
            ]
            result = subprocess.run(
                command,
                cwd=Path(temporary_directory),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Validation passed", result.stdout)
            self.assertEqual(expected_outputs, {path.name for path in output_dir.glob("*.jpg")})
            for output_name in expected_outputs:
                image = cv2.imread(str(output_dir / output_name))
                self.assertIsNotNone(image, output_name)
                self.assertGreater(image.size, 0, output_name)
            return result.stdout

    def test_channel_experiments(self) -> None:
        output = self.run_example(
            "channel_experiments/channel_experiments.py",
            {"blue_channel.jpg", "green_channel.jpg", "red_channel.jpg"},
        )
        self.assertIn("blue: contours=1", output)
        self.assertIn("green: contours=151", output)
        self.assertIn("red: contours=1321", output)

    def test_contour_approximations(self) -> None:
        output = self.run_example(
            "contour_approximations/contour_approx.py",
            {
                "image_thres1.jpg",
                "contours_none_image1.jpg",
                "contours_simple_image1.jpg",
                "contours_simple_image2.jpg",
                "contour_point_simple.jpg",
            },
        )
        self.assertIn("none_contours=85", output)
        self.assertIn("simple_contours=85", output)
        self.assertIn("image2_simple_contours=379", output)

    def test_contour_extraction(self) -> None:
        output = self.run_example(
            "contour_extraction/contour_extraction.py",
            {
                "contours_retr_list.jpg",
                "contours_retr_external.jpg",
                "contours_retr_ccomp.jpg",
                "contours_retr_tree.jpg",
            },
        )
        self.assertIn("LIST: contours=5", output)
        self.assertIn("EXTERNAL: contours=3", output)
        self.assertIn("CCOMP: contours=5", output)
        self.assertIn("TREE: contours=5", output)


if __name__ == "__main__":
    unittest.main()

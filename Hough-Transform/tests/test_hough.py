"""Regression tests for the Hough line and circle examples."""

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

import hough_circles  # noqa: E402
import hough_lines  # noqa: E402
import hough_utils  # noqa: E402


class HoughCoreTests(unittest.TestCase):
    def test_line_validation_recovers_semantic_shapes(self) -> None:
        result = hough_lines.run_validation()
        self.assertGreater(result["line_count"], 0)
        self.assertGreater(result["edge_pixels"], 0)
        self.assertGreater(result["synthetic_line_count"], 0)

    def test_circle_validation_recovers_known_geometry(self) -> None:
        result = hough_circles.run_validation()
        self.assertGreater(result["circle_count"], 0)
        self.assertGreater(result["synthetic_circle_count"], 0)

    def test_empty_line_result_has_stable_shape(self) -> None:
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        edges, lines = hough_utils.detect_lines(blank)
        self.assertEqual(lines.shape, (0, 4))
        self.assertEqual(int(cv2.countNonZero(edges)), 0)

    def test_empty_circle_result_has_stable_shape(self) -> None:
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        _, circles = hough_utils.detect_circles(
            blank,
            min_radius=10,
            max_radius=30,
        )
        self.assertEqual(circles.shape, (0, 3))

    def test_parameter_validation(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            hough_utils.detect_lines(image, canny_low=100, canny_high=50)
        with self.assertRaises(ValueError):
            hough_utils.detect_circles(image, min_radius=30, max_radius=10)


class HoughCliTests(unittest.TestCase):
    def _run(self, script: str, *arguments: str, cwd: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(PROJECT_DIR / script), *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_both_clis_run_headlessly_from_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "outputs"
            lines = self._run(
                "hough_lines.py",
                "--validate",
                "--no-display",
                "--output-dir",
                str(output_directory),
                cwd=temporary_directory,
            )
            circles = self._run(
                "hough_circles.py",
                "--validate",
                "--no-display",
                "--output-dir",
                str(output_directory),
                cwd=temporary_directory,
            )

            self.assertEqual(lines.returncode, 0, lines.stderr)
            self.assertEqual(circles.returncode, 0, circles.stderr)
            self.assertIn("VALIDATION PASSED", lines.stdout)
            self.assertIn("VALIDATION PASSED", circles.stdout)

            expected_outputs = {
                output_directory / "lanes-edges.png",
                output_directory / "lanes-lines.png",
                output_directory / "brown-eyes-blurred.png",
                output_directory / "brown-eyes-circles.png",
            }
            self.assertEqual(set(output_directory.glob("*.png")), expected_outputs)
            for output in expected_outputs:
                with self.subTest(output=output):
                    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
                    self.assertIsNotNone(image)
                    self.assertGreater(image.size, 0)

    def test_missing_input_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = self._run(
                "hough_lines.py",
                str(Path(temporary_directory) / "missing.jpg"),
                "--no-display",
                cwd=temporary_directory,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("could not read image", completed.stderr)


if __name__ == "__main__":
    unittest.main()

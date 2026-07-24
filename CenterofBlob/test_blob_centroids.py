#!/usr/bin/env python3
"""Unit and real-CLI regression tests for both blob-centroid examples."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import center_of_multiple_blob
import single_blob


# Resolve scripts and bundled assets independently of the test runner's cwd.
DIRECTORY = Path(__file__).resolve().parent


class SingleBlobTests(unittest.TestCase):
    """Exercise the reusable single-blob processing functions."""

    def test_dark_rectangle_centroid_and_mask(self) -> None:
        image = np.full((60, 80, 3), 255, dtype=np.uint8)
        image[20:40, 10:30] = 0

        annotated, centroid, mask = single_blob.process_image(image)

        self.assertEqual(centroid, (20, 30))
        self.assertEqual(cv2.countNonZero(mask), 400)
        self.assertFalse(np.shares_memory(image, annotated))

    def test_light_foreground(self) -> None:
        image = np.zeros((50, 70, 3), dtype=np.uint8)
        image[10:30, 30:50] = 255

        _, centroid, mask = single_blob.process_image(
            image, foreground="light"
        )

        self.assertEqual(centroid, (40, 20))
        self.assertEqual(cv2.countNonZero(mask), 400)

    def test_dark_foreground_measures_blob_not_background(self) -> None:
        image = np.full((30, 50, 3), 255, dtype=np.uint8)
        image[2:12, 3:13] = 0

        _, centroid, mask = single_blob.process_image(
            image, foreground="dark"
        )

        self.assertEqual(centroid, (8, 6))
        self.assertEqual(cv2.countNonZero(mask), 100)

    def test_empty_foreground_is_rejected(self) -> None:
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        mask = single_blob.create_binary_mask(image, foreground="dark")

        with self.assertRaisesRegex(ValueError, "No foreground pixels"):
            single_blob.find_centroid(mask)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            single_blob.create_binary_mask(None)

        image = np.zeros((5, 5, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            single_blob.create_binary_mask(image, threshold_value=256)
        with self.assertRaises(ValueError):
            single_blob.create_binary_mask(image, foreground="unknown")

    def test_bundled_circle_metrics(self) -> None:
        image = cv2.imread(str(DIRECTORY / "circle.png"), cv2.IMREAD_COLOR)

        _, centroid, mask = single_blob.process_image(image)

        self.assertEqual(centroid, single_blob.EXPECTED_CENTROID)
        self.assertEqual(
            cv2.countNonZero(mask),
            single_blob.EXPECTED_FOREGROUND_PIXELS,
        )


class MultipleBlobTests(unittest.TestCase):
    """Exercise contour filtering, measurement, and deterministic ordering."""

    def test_centroids_are_sorted_left_to_right(self) -> None:
        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        expected = [(25, 50), (80, 30), (130, 65)]
        for center in reversed(expected):
            cv2.circle(image, center, 10, (0, 0, 0), cv2.FILLED)

        _, blobs, _ = center_of_multiple_blob.process_image(
            image, min_area=100
        )

        self.assertEqual([blob[1] for blob in blobs], expected)

    def test_minimum_area_filters_small_contours(self) -> None:
        image = np.full((80, 80, 3), 255, dtype=np.uint8)
        cv2.circle(image, (20, 20), 2, (0, 0, 0), cv2.FILLED)
        cv2.circle(image, (55, 55), 10, (0, 0, 0), cv2.FILLED)

        _, blobs, _ = center_of_multiple_blob.process_image(
            image, min_area=100
        )

        self.assertEqual([blob[1] for blob in blobs], [(55, 55)])

    def test_zero_area_contour_is_skipped(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255

        self.assertEqual(
            center_of_multiple_blob.find_blob_centroids(mask), []
        )

    def test_invalid_mask_and_minimum_area_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary mask is empty"):
            center_of_multiple_blob.find_blob_centroids(None)

        mask = np.zeros((20, 20), dtype=np.uint8)
        for value in (-1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                center_of_multiple_blob.find_blob_centroids(
                    mask, min_area=value
                )

    def test_bundled_multiple_blob_metrics(self) -> None:
        image = cv2.imread(
            str(DIRECTORY / "multiple-blob.png"), cv2.IMREAD_COLOR
        )

        _, blobs, mask = center_of_multiple_blob.process_image(image)

        self.assertEqual(
            tuple(blob[1] for blob in blobs),
            center_of_multiple_blob.EXPECTED_CENTROIDS,
        )
        self.assertEqual(
            tuple(blob[2] for blob in blobs),
            center_of_multiple_blob.EXPECTED_AREAS,
        )
        self.assertEqual(
            cv2.countNonZero(mask),
            center_of_multiple_blob.EXPECTED_FOREGROUND_PIXELS,
        )


class CommandLineTests(unittest.TestCase):
    """Run the actual Python entry points from an unrelated directory."""

    def run_script(
        self, script_name: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(DIRECTORY / script_name), *arguments],
            cwd=self.working_directory,
            check=False,
            capture_output=True,
            text=True,
        )

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.working_directory = Path(self.temporary_directory.name)

    def assert_one_readable_output(
        self, output_directory: Path, expected_name: str, expected_shape
    ) -> None:
        self.assertEqual(
            sorted(path.name for path in output_directory.iterdir()),
            [expected_name],
        )
        output = cv2.imread(
            str(output_directory / expected_name), cv2.IMREAD_COLOR
        )
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, expected_shape)

    def test_single_default_input_validates_and_writes_exact_output(self) -> None:
        output_directory = self.working_directory / "single-output"

        result = self.run_script(
            "single_blob.py",
            "--no-display",
            "--validate",
            "--output-dir",
            str(output_directory),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Centroid: (112, 112)", result.stdout)
        self.assertIn("VALIDATION PASSED: single_blob", result.stdout)
        self.assertIn(f"OpenCV version: {cv2.__version__}", result.stdout)
        self.assert_one_readable_output(
            output_directory,
            single_blob.DEFAULT_OUTPUT_NAME,
            (225, 225, 3),
        )

    def test_multiple_default_input_validates_and_writes_exact_output(self) -> None:
        output_directory = self.working_directory / "multiple-output"

        result = self.run_script(
            "center_of_multiple_blob.py",
            "--no-display",
            "--validate",
            "--output-dir",
            str(output_directory),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Blob 6: centroid=(993, 116), area=14853.0", result.stdout
        )
        self.assertIn(
            "VALIDATION PASSED: center_of_multiple_blob", result.stdout
        )
        self.assertIn(f"OpenCV version: {cv2.__version__}", result.stdout)
        self.assert_one_readable_output(
            output_directory,
            center_of_multiple_blob.DEFAULT_OUTPUT_NAME,
            (236, 1089, 3),
        )

    def test_legacy_ipimage_alias_and_explicit_output(self) -> None:
        output_path = self.working_directory / "nested" / "result.png"

        result = self.run_script(
            "single_blob.py",
            "--ipimage",
            str(DIRECTORY / "circle.png"),
            "--no-display",
            "--output",
            str(output_path),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output_path.is_file())

    def test_missing_input_fails_clearly(self) -> None:
        missing = self.working_directory / "missing.png"

        result = self.run_script(
            "single_blob.py",
            "--input",
            str(missing),
            "--no-display",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("could not read input image", result.stderr)
        self.assertNotIn("VALIDATION PASSED", result.stdout)

    def test_validation_rejects_a_custom_input(self) -> None:
        custom_path = self.working_directory / "custom.png"
        custom = np.full((20, 20, 3), 255, dtype=np.uint8)
        cv2.circle(custom, (10, 10), 4, (0, 0, 0), cv2.FILLED)
        self.assertTrue(cv2.imwrite(str(custom_path), custom))

        result = self.run_script(
            "single_blob.py",
            "--input",
            str(custom_path),
            "--no-display",
            "--validate",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("--validate requires the bundled", result.stderr)
        self.assertNotIn("VALIDATION PASSED", result.stdout)

    def test_invalid_numeric_options_fail_during_parsing(self) -> None:
        invalid_threshold = self.run_script(
            "single_blob.py", "--threshold", "256", "--no-display"
        )
        invalid_area = self.run_script(
            "center_of_multiple_blob.py",
            "--min-area",
            "nan",
            "--no-display",
        )

        self.assertEqual(invalid_threshold.returncode, 2)
        self.assertIn(
            "--threshold must be between 0 and 255",
            invalid_threshold.stderr,
        )
        self.assertEqual(invalid_area.returncode, 2)
        self.assertIn(
            "--min-area must be a finite non-negative number",
            invalid_area.stderr,
        )


if __name__ == "__main__":
    unittest.main()

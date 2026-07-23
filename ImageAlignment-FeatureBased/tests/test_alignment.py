#!/usr/bin/env python3

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "align.py"
sys.path.insert(0, str(PROJECT_DIR))

from align import _align_images_impl, align_images, alignImages  # noqa: E402


EXPECTED_STATS = (500, 500, 500, 78, 51)
EXPECTED_OUTPUTS = {"aligned.jpg", "matches.jpg"}


class FeatureAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread(
            str(PROJECT_DIR / "scanned-form.jpg"),
            cv2.IMREAD_COLOR,
        )
        cls.reference = cv2.imread(
            str(PROJECT_DIR / "form.jpg"),
            cv2.IMREAD_COLOR,
        )
        if cls.image is None or cls.reference is None:
            raise RuntimeError("Could not load the bundled alignment fixtures")

    def _run_cli(self, working_directory, *arguments):
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *map(str, arguments)],
            cwd=working_directory,
            check=False,
            capture_output=True,
            text=True,
        )

    def _assert_successful_outputs(self, result, output_directory):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Alignment validation passed", result.stdout)
        self.assertIn(
            "Feature matches: total=500, retained=78, inliers=51",
            result.stdout,
        )
        self.assertEqual(
            {path.name for path in output_directory.iterdir()},
            EXPECTED_OUTPUTS,
        )

        aligned = cv2.imread(
            str(output_directory / "aligned.jpg"),
            cv2.IMREAD_COLOR,
        )
        matches = cv2.imread(
            str(output_directory / "matches.jpg"),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(aligned)
        self.assertIsNotNone(matches)
        self.assertEqual(aligned.shape, self.reference.shape)
        self.assertEqual(
            matches.shape,
            (
                max(self.image.shape[0], self.reference.shape[0]),
                self.image.shape[1] + self.reference.shape[1],
                3,
            ),
        )

    def test_align_images_fixture_contract(self):
        aligned, homography, stats = _align_images_impl(
            self.image,
            self.reference,
        )
        resized_input = cv2.resize(
            self.image,
            (self.reference.shape[1], self.reference.shape[0]),
        )
        before = float(np.mean(cv2.absdiff(resized_input, self.reference)))
        after = float(np.mean(cv2.absdiff(aligned, self.reference)))

        self.assertEqual(tuple(stats), EXPECTED_STATS)
        self.assertEqual(aligned.shape, self.reference.shape)
        self.assertEqual(homography.shape, (3, 3))
        self.assertTrue(np.isfinite(homography).all())
        self.assertNotEqual(float(np.linalg.det(homography)), 0.0)
        self.assertGreater(after / before, 0.23)
        self.assertLess(after / before, 0.30)

    def test_alignment_is_repeatable(self):
        _, first = align_images(self.image, self.reference)
        _, second = align_images(self.image, self.reference)
        corners = np.float32(
            [
                [0, 0],
                [self.image.shape[1] - 1, 0],
                [self.image.shape[1] - 1, self.image.shape[0] - 1],
                [0, self.image.shape[0] - 1],
            ]
        ).reshape(-1, 1, 2)
        first_projection = cv2.perspectiveTransform(corners, first)
        second_projection = cv2.perspectiveTransform(corners, second)
        np.testing.assert_allclose(
            first_projection,
            second_projection,
            rtol=0,
            atol=1e-6,
        )

    def test_legacy_alignImages_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            original_directory = Path.cwd()
            os.chdir(temporary_directory)
            try:
                aligned, homography = alignImages(self.image, self.reference)
            finally:
                os.chdir(original_directory)

            matches_path = Path(temporary_directory) / "matches.jpg"
            self.assertTrue(matches_path.is_file())
            self.assertGreater(matches_path.stat().st_size, 0)
            self.assertEqual(aligned.shape, self.reference.shape)
            self.assertEqual(homography.shape, (3, 3))

    def test_featureless_images_fail_cleanly(self):
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "features"):
            align_images(blank, blank)

    def test_cli_defaults_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            working_directory = temporary_path / "unrelated"
            output_directory = temporary_path / "nested" / "default-output"
            working_directory.mkdir()
            result = self._run_cli(
                working_directory,
                "--output-dir",
                output_directory,
                "--no-display",
                "--validate",
            )
            self._assert_successful_outputs(result, output_directory)

    def test_cli_explicit_inputs_and_nested_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            working_directory = temporary_path / "unrelated"
            output_directory = temporary_path / "nested" / "explicit-output"
            working_directory.mkdir()
            result = self._run_cli(
                working_directory,
                "--input",
                PROJECT_DIR / "scanned-form.jpg",
                "--reference",
                PROJECT_DIR / "form.jpg",
                "--output-dir",
                output_directory,
                "--no-display",
                "--validate",
            )
            self._assert_successful_outputs(result, output_directory)

    def test_cli_missing_inputs_fail_cleanly(self):
        cases = (
            (
                "--input",
                "missing-input.jpg",
                "Could not read input image",
            ),
            (
                "--reference",
                "missing-reference.jpg",
                "Could not read reference image",
            ),
        )
        for option, filename, expected_error in cases:
            with self.subTest(option=option):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    output_directory = temporary_path / "output"
                    result = self._run_cli(
                        temporary_path,
                        option,
                        temporary_path / filename,
                        "--output-dir",
                        output_directory,
                        "--no-display",
                        "--validate",
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(expected_error, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertFalse(output_directory.exists())

    def test_cli_unknown_argument(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = self._run_cli(
                temporary_directory,
                "--unknown",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unrecognized arguments: --unknown", result.stderr)


if __name__ == "__main__":
    unittest.main()

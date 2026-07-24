"""Numerical and public-CLI regressions for the camera-calibration tutorial."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


# Import the real project helper without depending on the test working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from calibration_utils import (  # noqa: E402
    calibrate_images,
    undistort_sample,
    validate_calibration,
)


# Stable bundled-fixture values shared by both exact OpenCV reference versions.
EXPECTED_IMAGE_COUNT = 41
EXPECTED_IMAGE_SIZE = (640, 480)
EXPECTED_RMS = 0.2603201
EXPECTED_REPROJECTION_RMSE = 0.26032
EXPECTED_CAMERA_MATRIX = np.array(
    [
        [503.5118, 0.0, 313.4135],
        [0.0, 503.1461, 243.0911],
        [0.0, 0.0, 1.0],
    ]
)
EXPECTED_OUTPUTS = {
    "undistorted_direct.jpg",
    "undistorted_remap.jpg",
}


class CameraCalibrationTest(unittest.TestCase):
    """Exercise shared numerics and both shipped Python programs."""

    @classmethod
    def setUpClass(cls) -> None:
        # Calibrate once because all direct helper tests use the same fixture.
        cls.calibration = calibrate_images(
            str(PROJECT_DIR / "images" / "*.jpg"),
            display=False,
        )

    def test_bundled_calibration_baseline(self) -> None:
        """All images, dimensions, errors, and intrinsics match the oracle."""

        result = self.calibration
        validate_calibration(result)
        self.assertEqual(EXPECTED_IMAGE_COUNT, len(result.image_paths))
        self.assertEqual(EXPECTED_IMAGE_COUNT, result.detected_images)
        self.assertEqual(EXPECTED_IMAGE_SIZE, result.image_size)
        self.assertAlmostEqual(EXPECTED_RMS, result.rms_error, delta=0.01)
        self.assertAlmostEqual(
            EXPECTED_REPROJECTION_RMSE,
            result.reprojection_rmse,
            delta=0.01,
        )
        np.testing.assert_allclose(
            EXPECTED_CAMERA_MATRIX,
            result.camera_matrix,
            rtol=0.01,
            atol=0.01,
        )

    def test_undistortion_methods_agree(self) -> None:
        """Both workflows produce readable same-sized images with close pixels."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            direct, remapped, difference = undistort_sample(
                self.calibration,
                self.calibration.image_paths[0],
                output_dir,
                display=False,
            )
            self.assertEqual((480, 640, 3), direct.shape)
            self.assertEqual(direct.shape, remapped.shape)
            # OpenCV 5.0 differs more than 4.14 but remains below 0.1 intensity.
            self.assertLess(difference, 0.1)
            self.assertEqual(
                EXPECTED_OUTPUTS,
                {path.name for path in output_dir.iterdir()},
            )
            for filename in EXPECTED_OUTPUTS:
                output = cv2.imread(str(output_dir / filename))
                self.assertIsNotNone(output, filename)
                self.assertEqual(direct.shape, output.shape)
                self.assertGreater(int(output.max()), int(output.min()))

    def test_missing_images_fail_cleanly(self) -> None:
        """The reusable helper rejects an empty glob before calibration."""

        with self.assertRaises(FileNotFoundError):
            calibrate_images(
                str(PROJECT_DIR / "images" / "missing-*.jpg"),
                display=False,
            )

    def test_real_entry_points_from_unrelated_directory(self) -> None:
        """Both scripts validate headlessly without a source-relative cwd."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            unrelated_directory = Path(temporary_directory)
            calibration_process = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "cameraCalibration.py"),
                    "--no-display",
                    "--validate",
                ],
                cwd=unrelated_directory,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                0,
                calibration_process.returncode,
                calibration_process.stderr,
            )
            self.assertIn("images=41, detections=41", calibration_process.stdout)
            self.assertIn("Rotation vectors:", calibration_process.stdout)
            self.assertIn("Translation vectors:", calibration_process.stdout)
            self.assertIn("[40]", calibration_process.stdout)
            self.assertIn("Validation passed", calibration_process.stdout)

            output_dir = unrelated_directory / "generated"
            undistortion_process = subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_DIR
                        / "cameraCalibrationWithUndistortion.py"
                    ),
                    "--images",
                    str(PROJECT_DIR / "images" / "*.jpg"),
                    "--sample",
                    str(PROJECT_DIR / "images" / "image_10.jpg"),
                    "--no-display",
                    "--validate",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=unrelated_directory,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                0,
                undistortion_process.returncode,
                undistortion_process.stderr,
            )
            self.assertIn("images=41, detections=41", undistortion_process.stdout)
            self.assertIn("Rotation vectors:", undistortion_process.stdout)
            self.assertIn("Translation vectors:", undistortion_process.stdout)
            self.assertIn("[40]", undistortion_process.stdout)
            self.assertIn("Validation passed", undistortion_process.stdout)
            self.assertEqual(
                EXPECTED_OUTPUTS,
                {path.name for path in output_dir.iterdir()},
            )
            for filename in EXPECTED_OUTPUTS:
                generated = cv2.imread(str(output_dir / filename))
                self.assertIsNotNone(generated, filename)
                self.assertEqual((480, 640, 3), generated.shape)

    def test_calibration_cli_reports_missing_input(self) -> None:
        """The public CLI returns a concise nonzero error for a missing glob."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            process = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "cameraCalibration.py"),
                    "--images",
                    str(Path(temporary_directory) / "missing-*.jpg"),
                    "--no-display",
                ],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(1, process.returncode)
        self.assertIn("Error: No calibration images matched:", process.stderr)
        self.assertNotIn("Traceback", process.stderr)

    def test_undistortion_cli_rejects_unknown_option(self) -> None:
        """The second entry point retains argparse's explicit option failure."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            process = subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_DIR
                        / "cameraCalibrationWithUndistortion.py"
                    ),
                    "--not-an-option",
                ],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(2, process.returncode)
        self.assertIn("unrecognized arguments: --not-an-option", process.stderr)


if __name__ == "__main__":
    unittest.main()

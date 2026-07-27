"""Regression tests for the rotation-matrix tutorial."""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import rotm2euler  # noqa: E402


class RotationConversionTests(unittest.TestCase):
    def test_known_axis_rotations_round_trip(self) -> None:
        cases = (
            np.zeros(3),
            np.array([math.pi / 2.0, 0.0, 0.0]),
            np.array([0.0, math.pi / 2.0, 0.0]),
            np.array([0.0, -math.pi / 2.0, 0.0]),
            np.array([0.3, -0.7, 1.2]),
        )
        for angles in cases:
            with self.subTest(angles=angles):
                matrix = rotm2euler.euler_angles_to_rotation_matrix(angles)
                recovered = rotm2euler.rotation_matrix_to_euler_angles(matrix)
                rebuilt = rotm2euler.euler_angles_to_rotation_matrix(recovered)
                np.testing.assert_allclose(rebuilt, matrix, atol=1e-10, rtol=0.0)

    def test_near_gimbal_lock_round_trip(self) -> None:
        for pitch in (math.pi / 2.0 - 1e-10, -math.pi / 2.0 + 1e-10):
            matrix = rotm2euler.euler_angles_to_rotation_matrix([0.4, pitch, -1.2])
            recovered = rotm2euler.rotation_matrix_to_euler_angles(matrix)
            rebuilt = rotm2euler.euler_angles_to_rotation_matrix(recovered)
            np.testing.assert_allclose(rebuilt, matrix, atol=1e-9, rtol=0.0)

    def test_rejects_reflections_and_malformed_inputs(self) -> None:
        self.assertFalse(rotm2euler.is_rotation_matrix(np.diag([1.0, 1.0, -1.0])))
        self.assertFalse(rotm2euler.is_rotation_matrix(np.eye(4)))
        self.assertFalse(rotm2euler.is_rotation_matrix(np.full((3, 3), np.nan)))
        with self.assertRaises(ValueError):
            rotm2euler.rotation_matrix_to_euler_angles(np.diag([1.0, 1.0, -1.0]))

    def test_deterministic_validation(self) -> None:
        result = rotm2euler.run_validation()
        self.assertEqual(result["cases"], 518)
        self.assertLessEqual(float(result["max_matrix_error"]), 1e-9)

    def test_cli_runs_from_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT_DIR / "rotm2euler.py"),
                    "--validate",
                ],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("VALIDATION PASSED", completed.stdout)

    def test_cli_reports_invalid_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(PROJECT_DIR / "rotm2euler.py"),
                    "--matrix",
                    "1",
                    "0",
                    "0",
                    "0",
                    "1",
                    "0",
                    "0",
                    "0",
                    "-1",
                ],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("proper rotation matrix", completed.stderr)


if __name__ == "__main__":
    unittest.main()

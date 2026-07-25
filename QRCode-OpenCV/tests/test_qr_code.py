"""Regression tests for the real QR-code example entry point."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "qrCodeOpencv.py"
COMPARISON_SCRIPT_PATH = PROJECT_DIR / "zbar-opencv-comparison.py"


class QRCodeRegressionTest(unittest.TestCase):
    """Exercise the CLI outside the project working directory."""

    def test_bundled_qr_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_dir = temporary_path / "outputs"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--no-display",
                    "--validate",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=temporary_path,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn("Decoded data: http://LearnOpenCV.com", completed.stdout)
            self.assertIn("VALIDATION PASSED", completed.stdout)

            expected_outputs = {
                "qr-code-annotated.png",
                "qr-code-rectified.png",
            }
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                expected_outputs,
            )
            for output_name in expected_outputs:
                image = cv2.imread(str(output_dir / output_name))
                self.assertIsNotNone(image)
                self.assertGreater(image.size, 0)

    def test_missing_input_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.png"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--input",
                    str(missing_path),
                    "--no-display",
                ],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Unable to read input image", completed.stdout)

    def test_optional_zbar_comparison(self) -> None:
        try:
            import pyzbar.pyzbar  # noqa: F401
        except (ImportError, OSError):
            self.skipTest("pyzbar and the system zbar library are optional")

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_path = temporary_path / "comparison.png"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(COMPARISON_SCRIPT_PATH),
                    "--no-display",
                    "--validate",
                    "--output",
                    str(output_path),
                ],
                cwd=temporary_path,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn("VALIDATION PASSED", completed.stdout)
            self.assertIsNotNone(cv2.imread(str(output_path)))


if __name__ == "__main__":
    unittest.main()

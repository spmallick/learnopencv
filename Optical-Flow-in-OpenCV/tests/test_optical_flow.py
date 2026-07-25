"""End-to-end regression tests for all Python optical-flow modes."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "demo.py"
ALGORITHMS = ("farneback", "lucaskanade", "lucaskanade_dense", "rlof")


class OpticalFlowRegressionTest(unittest.TestCase):
    """Run each real CLI from outside its project directory."""

    def test_all_algorithms_headlessly(self) -> None:
        if not hasattr(cv2, "optflow"):
            self.fail("The regression matrix requires opencv-contrib-python")

        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_path = Path(temporary_directory)
                    output_dir = temporary_path / "outputs"
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(SCRIPT_PATH),
                            "--algorithm",
                            algorithm,
                            "--no-display",
                            "--validate",
                            "--max-frames",
                            "3",
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
                    self.assertIn("Frame pairs processed: 3", completed.stdout)
                    self.assertIn("VALIDATION PASSED", completed.stdout)
                    output_files = list(output_dir.glob("*.png"))
                    self.assertEqual(len(output_files), 1)
                    image = cv2.imread(str(output_files[0]))
                    self.assertIsNotNone(image)
                    self.assertGreater(image.size, 0)

    def test_missing_video_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.mp4"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--algorithm",
                    "farneback",
                    "--video",
                    str(missing_path),
                    "--no-display",
                ],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Unable to open input video", completed.stdout)


if __name__ == "__main__":
    unittest.main()

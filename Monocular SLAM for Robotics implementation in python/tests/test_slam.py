"""Regression tests for the real monocular-SLAM entry point."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "main.py"


class MonocularSlamRegressionTest(unittest.TestCase):
    """Run headlessly from outside the project directory."""

    def test_bundled_video_produces_map_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_dir = temporary_path / "outputs"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--no-display",
                    "--validate",
                    "--max-frames",
                    "12",
                    "--width",
                    "640",
                    "--focal-length",
                    "300",
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
            self.assertIn("VALIDATION PASSED", completed.stdout)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "slam-feature-tracks.png",
                    "slam-map.ply",
                    "slam-trajectory.png",
                },
            )
            for image_name in (
                "slam-feature-tracks.png",
                "slam-trajectory.png",
            ):
                image = cv2.imread(str(output_dir / image_name))
                self.assertIsNotNone(image)
                self.assertGreater(image.size, 0)
            self.assertTrue(
                (output_dir / "slam-map.ply")
                .read_text(encoding="utf-8")
                .startswith("ply\n")
            )

    def test_missing_video_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.mp4"
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
            self.assertIn("Unable to open input video", completed.stdout)


if __name__ == "__main__":
    unittest.main()

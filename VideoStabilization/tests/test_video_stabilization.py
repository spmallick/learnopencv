"""Regression tests for the real Python video-stabilization entry points."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import video_stabilization as stabilizer  # noqa: E402


class VideoStabilizationTests(unittest.TestCase):
    def test_moving_average_keeps_shape_and_edges(self) -> None:
        curve = np.array([0.0, 3.0, 6.0])
        actual = stabilizer.moving_average(curve, radius=1)
        np.testing.assert_allclose(actual, [1.0, 3.0, 5.0])

    def test_cli_runs_from_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "input.avi"
            output_dir = temporary_path / "result"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20.0,
                (160, 120),
            )
            self.assertTrue(writer.isOpened())
            for index in range(18):
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                cv2.rectangle(
                    frame,
                    (25 + index, 25),
                    (75 + index, 75),
                    (255, 255, 255),
                    -1,
                )
                cv2.circle(frame, (110 + index // 2, 85), 12, (0, 255, 0), -1)
                writer.write(frame)
            writer.release()

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "video_stabilization.py"),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                    "--no-display",
                    "--validate",
                ],
                cwd=temporary_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn(
                "VALIDATION PASSED: 17 frames, 320x120", result.stdout
            )
            output_path = output_dir / "video_out.mp4"
            self.assertEqual(list(output_dir.iterdir()), [output_path])
            self.assertTrue(output_path.is_file())
            capture = cv2.VideoCapture(str(output_path))
            decoded_frames = 0
            first_shape = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if first_shape is None:
                    first_shape = frame.shape[:2]
                decoded_frames += 1
            capture.release()
            self.assertEqual(decoded_frames, 17)
            self.assertEqual(first_shape, (120, 320))

    def test_bundled_cli_has_stable_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            output_dir = temporary_path / "bundled-result"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "video_stabilization.py"),
                    "--output-dir",
                    str(output_dir),
                    "--no-display",
                    "--validate",
                ],
                cwd=temporary_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn(
                "VALIDATION PASSED: 1321 frames, 1440x480", result.stdout
            )
            match = re.search(
                r"mean tracked points ([0-9]+(?:\.[0-9]+)?)",
                result.stdout,
            )
            self.assertIsNotNone(match)
            mean_tracked = float(match.group(1))
            self.assertGreater(mean_tracked, 100.0)
            self.assertLess(mean_tracked, 120.0)
            output_path = output_dir / "video_out.mp4"
            self.assertEqual(list(output_dir.iterdir()), [output_path])
            capture = cv2.VideoCapture(str(output_path))
            decoded_frames = 0
            first_shape = None
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if first_shape is None:
                    first_shape = frame.shape[:2]
                decoded_frames += 1
            capture.release()
            self.assertEqual(decoded_frames, 1321)
            self.assertEqual(first_shape, (480, 1440))

    def test_missing_input_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "video_stabilization.py"),
                    "--input",
                    str(Path(temporary) / "missing.mp4"),
                    "--no-display",
                ],
                cwd=temporary,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Could not open input video", result.stdout)

    def test_cli_refuses_to_overwrite_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "do-not-overwrite.avi"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20.0,
                (80, 60),
            )
            self.assertTrue(writer.isOpened())
            for index in range(4):
                frame = np.full((60, 80, 3), index * 40, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            original_bytes = input_path.read_bytes()

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "video_stabilization.py"),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(temporary_path),
                    "--output-name",
                    input_path.name,
                    "--no-display",
                ],
                cwd=temporary_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "input and output videos must use different paths",
                result.stdout,
            )
            self.assertEqual(input_path.read_bytes(), original_bytes)

    def test_cli_reports_output_directory_error_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            input_path = temporary_path / "short-input.avi"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20.0,
                (80, 60),
            )
            self.assertTrue(writer.isOpened())
            for index in range(4):
                frame = np.zeros((60, 80, 3), dtype=np.uint8)
                cv2.circle(
                    frame,
                    (20 + index * 3, 30),
                    8,
                    (255, 255, 255),
                    -1,
                )
                writer.write(frame)
            writer.release()

            blocked_output_dir = temporary_path / "not-a-directory"
            blocked_output_dir.write_text("blocking file", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "video_stabilization.py"),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(blocked_output_dir),
                    "--no-display",
                ],
                cwd=temporary_path,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ERROR:", result.stdout)
            self.assertNotIn("Traceback", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()

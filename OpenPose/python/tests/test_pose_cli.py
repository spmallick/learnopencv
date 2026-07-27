"""End-to-end regression tests for the real MediaPipe Pose CLI entry points."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL = Path(
    os.environ.get(
        "OPENPOSE_MODEL_PATH",
        PROJECT_DIR / "models" / "pose_estimation_mediapipe_2023mar.onnx",
    )
).resolve()
sys.path.insert(0, str(PROJECT_DIR))
from download_models import verify as verify_model  # noqa: E402


class PoseCliTests(unittest.TestCase):
    """Exercise defaults from an unrelated working directory."""

    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL.is_file():
            raise FileNotFoundError(
                f"Model missing: {MODEL}. Run download_models.py first."
            )

    def run_cli(self, script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as working_directory:
            return subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / script),
                    "--model",
                    str(MODEL),
                    *arguments,
                ],
                cwd=working_directory,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_pinned_model_checksum(self) -> None:
        verify_model(MODEL)

    def test_image_cli_writes_readable_same_size_output(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            completed = self.run_cli(
                "OpenPoseImage.py",
                "--output-dir",
                output_directory,
                "--no-display",
                "--validate",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "VALIDATION PASSED: landmarks=33 visible=33 edges=35",
                completed.stdout,
            )
            source = cv2.imread(str(PROJECT_DIR / "single.jpeg"))
            output = cv2.imread(str(Path(output_directory) / "pose-image.jpg"))
            self.assertIsNotNone(source)
            self.assertIsNotNone(output)
            self.assertEqual(output.shape, source.shape)

    def test_video_cli_preserves_dimensions_and_requested_frames(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            completed = self.run_cli(
                "OpenPoseVideo.py",
                "--output-dir",
                output_directory,
                "--max-frames",
                "2",
                "--no-display",
                "--validate",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("VALIDATION PASSED: frames=2", completed.stdout)
            output_path = Path(output_directory) / "pose-video.avi"
            capture = cv2.VideoCapture(str(output_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 576)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            capture.release()

    def test_missing_input_fails_clearly(self) -> None:
        completed = self.run_cli(
            "OpenPoseImage.py",
            "--input",
            "/definitely/missing/pose-input.jpg",
            "--no-display",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Could not read input image", completed.stderr)


if __name__ == "__main__":
    unittest.main()

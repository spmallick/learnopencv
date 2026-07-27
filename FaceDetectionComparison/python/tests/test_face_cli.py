"""End-to-end tests for dependency-free YuNet image/video entry points."""

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
        "YUNET_MODEL_PATH",
        PROJECT_DIR / "models" / "face_detection_yunet_2026may.onnx",
    )
).resolve()
sys.path.insert(0, str(PROJECT_DIR))
from download_models import verify as verify_model  # noqa: E402


class FaceCliTests(unittest.TestCase):
    """Run real scripts from unrelated temporary directories."""

    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL.is_file():
            raise FileNotFoundError(
                f"Model missing: {MODEL}. Run download_models.py first."
            )

    def run_cli(
        self,
        script: str,
        *arguments: str,
    ) -> subprocess.CompletedProcess[str]:
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

    def create_image_fixture(self, directory: Path) -> Path:
        """Decode the bundled video's first frame into a temporary JPEG."""

        capture = cv2.VideoCapture(str(PROJECT_DIR / "videos" / "baby.mp4"))
        has_frame, frame = capture.read()
        capture.release()
        self.assertTrue(has_frame)
        image_path = directory / "baby-first-frame.jpg"
        self.assertTrue(cv2.imwrite(str(image_path), frame))
        return image_path

    def test_primary_image_cli_detects_two_faces_and_preserves_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = self.create_image_fixture(root)
            output_directory = root / "output"
            completed = self.run_cli(
                "face_detection_opencv_dnn.py",
                "--input",
                str(image_path),
                "--mode",
                "image",
                "--output-dir",
                str(output_directory),
                "--no-display",
                "--validate",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("VALIDATION PASSED: mode=image faces=2", completed.stdout)
            source = cv2.imread(str(image_path))
            output = cv2.imread(str(output_directory / "yunet-image.jpg"))
            self.assertIsNotNone(output)
            self.assertEqual(output.shape, source.shape)

    def test_primary_video_cli_preserves_size_and_two_frames(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            completed = self.run_cli(
                "face_detection_opencv_dnn.py",
                "--output-dir",
                output_directory,
                "--max-frames",
                "2",
                "--no-display",
                "--validate",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("VALIDATION PASSED: mode=video frames=2", completed.stdout)
            self.assertIn(
                "FACE VIDEO RESULT: detector=YuNet frames=2 total_faces=4",
                completed.stdout,
            )
            capture = cv2.VideoCapture(
                str(Path(output_directory) / "yunet-video.avi")
            )
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 720)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            capture.release()

    def test_default_comparison_is_yunet_only_and_correctly_sized(self) -> None:
        with tempfile.TemporaryDirectory() as output_directory:
            completed = self.run_cli(
                "run-all.py",
                "--output-dir",
                output_directory,
                "--max-frames",
                "2",
                "--no-display",
                "--validate",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("panels=1 frames=2 size=720x720", completed.stdout)
            self.assertIn(
                "COMPARISON VIDEO RESULT: frames=2 YuNet=4",
                completed.stdout,
            )
            capture = cv2.VideoCapture(
                str(Path(output_directory) / "comparison-video.avi")
            )
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 720)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 720)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            capture.release()

    def test_missing_input_fails_clearly(self) -> None:
        completed = self.run_cli(
            "face_detection_opencv_dnn.py",
            "--input",
            "/definitely/missing/face-video.mp4",
            "--mode",
            "video",
            "--no-display",
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("Could not open input video", completed.stderr)


if __name__ == "__main__":
    unittest.main()

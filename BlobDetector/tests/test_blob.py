from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import blob  # noqa: E402


class BlobDetectorTests(unittest.TestCase):
    def test_sample_image_has_expected_blob_count(self) -> None:
        image = blob.read_grayscale(PROJECT_DIR / "blob.jpg")
        keypoints = blob.detect_blobs(image)

        self.assertEqual(len(keypoints), 16)

    def test_detector_rejects_color_input(self) -> None:
        color_image = np.zeros((32, 32, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "grayscale"):
            blob.detect_blobs(color_image)

    def test_cli_writes_color_visualization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.png"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "blob.py"),
                    "--input",
                    str(PROJECT_DIR / "blob.jpg"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Detected 16 blobs.", completed.stdout)
            result = cv2.imread(str(output), cv2.IMREAD_COLOR)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (480, 480, 3))


if __name__ == "__main__":
    unittest.main()

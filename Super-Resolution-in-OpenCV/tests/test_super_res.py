from __future__ import annotations

import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import download_model  # noqa: E402
import super_res  # noqa: E402


class SuperResolutionTests(unittest.TestCase):
    def test_rejects_invalid_algorithm_scale_pair(self):
        with self.assertRaisesRegex(ValueError, "not supported"):
            super_res.validate_model_configuration("lapsrn", 3)

    def test_rejects_missing_model(self):
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        with self.assertRaisesRegex(FileNotFoundError, "Model not found"):
            super_res.upscale_image(image, Path("does-not-exist.pb"))

    @unittest.skipUnless(
        os.environ.get("OPENCV_SUPERRES_MODEL"),
        "Set OPENCV_SUPERRES_MODEL to run model inference.",
    )
    def test_pinned_model_checksum(self):
        model = Path(os.environ["OPENCV_SUPERRES_MODEL"])
        self.assertEqual(download_model.sha256(model), download_model.MODEL_SHA256)
        self.assertEqual(model.stat().st_size, 100_323)

    @unittest.skipUnless(
        os.environ.get("OPENCV_SUPERRES_MODEL"),
        "Set OPENCV_SUPERRES_MODEL to test verified model installation.",
    )
    def test_downloader_installs_verified_bytes_atomically(self):
        model_bytes = Path(os.environ["OPENCV_SUPERRES_MODEL"]).read_bytes()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "models" / "ESPCN_x4.pb"
            with mock.patch.object(
                download_model,
                "urlopen",
                return_value=io.BytesIO(model_bytes),
            ):
                installed = download_model.download_model(output)
            self.assertEqual(installed, output)
            self.assertEqual(
                download_model.sha256(installed), download_model.MODEL_SHA256
            )

    @unittest.skipUnless(
        os.environ.get("OPENCV_SUPERRES_MODEL"),
        "Set OPENCV_SUPERRES_MODEL to run model inference.",
    )
    def test_espcn_x4_inference(self):
        model = Path(os.environ["OPENCV_SUPERRES_MODEL"])
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        image[:, :8] = (0, 127, 255)
        result = super_res.upscale_image(image, model, "espcn", 4)
        self.assertEqual(result.shape, (48, 64, 3))
        self.assertEqual(result.dtype, np.uint8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "upscaled.png"
            self.assertTrue(cv2.imwrite(str(output), result))
            self.assertIsNotNone(cv2.imread(str(output), cv2.IMREAD_COLOR))


if __name__ == "__main__":
    unittest.main()

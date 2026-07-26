from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2 as cv
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import augmented_reality_with_aruco as ar  # noqa: E402
import generate_aruco_markers as generator  # noqa: E402


class ArucoAugmentedRealityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = cv.imread(str(PROJECT_DIR / "test.jpg"), cv.IMREAD_COLOR)
        cls.overlay = cv.imread(
            str(PROJECT_DIR / "new_scenery.jpg"), cv.IMREAD_COLOR
        )
        cls.detector = ar.create_detector()

    def test_bundled_image_detects_required_markers_and_augments(self):
        augmented, did_augment, detected_ids = ar.augment_frame(
            self.frame, self.overlay, self.detector
        )
        self.assertTrue(did_augment)
        self.assertEqual(detected_ids, [23, 25, 30, 33])
        self.assertEqual(augmented.shape, self.frame.shape)
        difference = cv.absdiff(augmented, self.frame)
        self.assertGreater(np.count_nonzero(difference), 10_000)

    def test_missing_markers_preserves_frame(self):
        blank = np.full((240, 320, 3), 255, dtype=np.uint8)
        augmented, did_augment, detected_ids = ar.augment_frame(
            blank, self.overlay, self.detector
        )
        self.assertFalse(did_augment)
        self.assertEqual(detected_ids, [])
        np.testing.assert_array_equal(augmented, blank)

    def test_generated_marker_round_trip(self):
        marker = generator.generate_marker(marker_id=33, size=200)
        canvas = np.full((240, 240), 255, dtype=np.uint8)
        canvas[20:220, 20:220] = marker
        _, marker_ids, _ = self.detector.detectMarkers(canvas)
        self.assertIsNotNone(marker_ids)
        self.assertIn(33, marker_ids.reshape(-1).tolist())

    def test_image_cli_writes_augmented_only_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "aruco-result.jpg"
            exit_code = ar.main(
                [
                    "--image",
                    str(PROJECT_DIR / "test.jpg"),
                    "--overlay",
                    str(PROJECT_DIR / "new_scenery.jpg"),
                    "--output",
                    str(output),
                    "--augmented-only",
                    "--strict",
                ]
            )
            self.assertEqual(exit_code, 0)
            result = cv.imread(str(output), cv.IMREAD_COLOR)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, self.frame.shape)


if __name__ == "__main__":
    unittest.main()

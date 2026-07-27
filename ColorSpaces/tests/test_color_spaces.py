"""Regression tests for the Color Spaces tutorial programs."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import color_spaces  # noqa: E402
import dataAnalysis  # noqa: E402
import interactiveColorDetect  # noqa: E402
import interactiveColorSegment  # noqa: E402


class ColorSpaceCoreTests(unittest.TestCase):
    def test_known_pixel_conversions(self) -> None:
        self.assertEqual(
            color_spaces.convert_pixel((40, 158, 16)),
            {
                "BGR": (40, 158, 16),
                "HSV": (65, 229, 158),
                "YCrCb": (102, 67, 93),
                "Lab": (145, 71, 177),
            },
        )

    def test_bundled_asset_inventory_is_readable(self) -> None:
        metrics = color_spaces.run_core_validation()
        self.assertEqual(metrics, {"cube_images": 10, "piece_images": 56})

    def test_hsv_hue_wrap_includes_red_but_not_green(self) -> None:
        image = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
        mask = color_spaces.threshold_mask(
            image,
            "HSV",
            (170, 100, 100),
            (10, 255, 255),
        )
        np.testing.assert_array_equal(mask, np.array([[255, 0]], dtype=np.uint8))

    def test_pixel_panel_rejects_out_of_bounds_coordinates(self) -> None:
        image = color_spaces.read_bgr(color_spaces.resolve_input_path(None))
        with self.assertRaises(ValueError):
            interactiveColorDetect.render_pixel_panel(image, image.shape[1], 0)

    def test_default_segmentation_is_nonempty(self) -> None:
        metrics = interactiveColorSegment.run_validation()
        self.assertGreater(metrics["foreground_pixels"], 0)

    def test_density_analysis_has_consistent_channel_lengths(self) -> None:
        files = dataAnalysis.sample_files(PROJECT_DIR / "pieces", "yellow")
        channels = dataAnalysis.collect_channels(files)
        lengths = {channel.size for channel in channels.values()}
        self.assertEqual(len(files), 8)
        self.assertEqual(len(lengths), 1)
        self.assertGreater(next(iter(lengths)), 0)


class ColorSpaceCliTests(unittest.TestCase):
    def _run(self, script: str, *arguments: str, cwd: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        return subprocess.run(
            [sys.executable, "-B", str(PROJECT_DIR / script), *arguments],
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_all_python_clis_run_headlessly_from_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            detect_output = temporary_path / "detect.png"
            segment_output = temporary_path / "segment"
            density_output = temporary_path / "density.png"

            detect = self._run(
                "interactiveColorDetect.py",
                "--validate",
                "--no-display",
                "--output",
                str(detect_output),
                cwd=temporary_directory,
            )
            segment = self._run(
                "interactiveColorSegment.py",
                "--validate",
                "--no-display",
                "--output-dir",
                str(segment_output),
                cwd=temporary_directory,
            )
            density = self._run(
                "dataAnalysis.py",
                "--validate",
                "--output",
                str(density_output),
                cwd=temporary_directory,
            )

            self.assertEqual(detect.returncode, 0, detect.stderr)
            self.assertEqual(segment.returncode, 0, segment.stderr)
            self.assertEqual(density.returncode, 0, density.stderr)
            self.assertIn("VALIDATION PASSED", detect.stdout)
            self.assertIn("VALIDATION PASSED", segment.stdout)
            self.assertIn("VALIDATION PASSED", density.stdout)

            expected_outputs = {
                detect_output,
                density_output,
                segment_output / "rub00-hsv-mask.png",
                segment_output / "rub00-hsv-result.png",
            }
            for output in expected_outputs:
                with self.subTest(output=output):
                    self.assertTrue(output.is_file())
                    image = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
                    self.assertIsNotNone(image)
                    self.assertGreater(image.size, 0)

    def test_missing_input_reports_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = self._run(
                "interactiveColorDetect.py",
                "--input",
                str(Path(temporary_directory) / "missing.jpg"),
                "--no-display",
                cwd=temporary_directory,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("could not read image", completed.stderr)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for alignment direction, rejection, and real CLI behavior."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


# Import the delivered example regardless of the test runner's working directory.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import robust_image_alignment as alignment


def textured_image(seed: int = 4, shape: tuple[int, int] = (320, 400)) -> np.ndarray:
    """Create original test pixels without depending on a downloadable photograph."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (*shape, 3), np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0.6)
    cv2.putText(image, "REFERENCE 42", (25, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 3)
    cv2.rectangle(image, (40, 180), (170, 285), (200, 40, 20), 3)
    return image


class GeometryTests(unittest.TestCase):
    def test_recovers_known_transform_despite_outliers(self) -> None:
        rng = np.random.default_rng(11)
        reference_points = rng.uniform([30, 30], [370, 290], (100, 2)).astype(np.float32)
        reference_to_moving = np.array([[0.94, -0.08, 30], [0.04, 0.96, 8], [0.00012, -0.00008, 1]])
        moving_points = alignment.project_points(reference_points, reference_to_moving)
        # Thirty deliberately wrong correspondences exercise robust estimation.
        moving_points[-30:] = rng.uniform([0, 0], [399, 319], (30, 2))
        matches = alignment.MatchResult(reference_points, moving_points)
        for estimator in ("ransac", "magsac"):
            with self.subTest(estimator=estimator):
                result = alignment.estimate_alignment(matches, (320, 400), (320, 400), estimator=estimator)
                self.assertTrue(result.accepted, result.reasons)
                # Check independent corners, rather than merely the fitted points.
                expected = alignment.project_points(alignment.image_corners((320, 400)), reference_to_moving)
                np.testing.assert_allclose(result.projected_reference_quad, expected, atol=0.05)
                self.assertGreaterEqual(result.num_inliers, 70)
                recovered = alignment.project_points(moving_points[:70], result.H)
                np.testing.assert_allclose(recovered, reference_points[:70], atol=0.05)

    def test_small_target_in_large_scene_is_not_rejected_for_scene_coverage(self) -> None:
        reference_points = np.array([(x, y) for x in range(15, 190, 30) for y in range(15, 150, 25)], np.float32)
        reference_to_scene = np.array([[0.5, 0, 500], [0, 0.5, 300], [0, 0, 1]], np.float64)
        matches = alignment.MatchResult(reference_points, alignment.project_points(reference_points, reference_to_scene))
        result = alignment.estimate_alignment(matches, (160, 200), (1000, 1200))
        self.assertTrue(result.accepted, result.reasons)
        self.assertLess(result.moving_coverage, 0.01)
        self.assertGreater(result.reference_coverage, 0.5)

    def test_rejects_no_matches_and_collinear_points(self) -> None:
        cases = [np.empty((0, 2), np.float32), np.array([(x, 10) for x in range(20)], np.float32)]
        for points in cases:
            with self.subTest(count=len(points)):
                result = alignment.estimate_alignment(alignment.MatchResult(points, points), (100, 100), (100, 100))
                self.assertFalse(result.accepted)
                self.assertIsNone(result.H)
                self.assertTrue(result.reasons)

    def test_rejects_independent_random_correspondences(self) -> None:
        rng = np.random.default_rng(31)
        matches = alignment.MatchResult(rng.uniform(0, 299, (80, 2)), rng.uniform(0, 299, (80, 2)))
        result = alignment.estimate_alignment(matches, (300, 300), (300, 300))
        self.assertFalse(result.accepted)
        self.assertTrue(result.reasons)
        # Rejected transformations must not silently generate a trusted output.
        with self.assertRaisesRegex(ValueError, "rejected"):
            alignment.warp_alignment(np.zeros((300, 300, 3), np.uint8), result, (300, 300))

    def test_reference_spatial_coverage_rejects_small_matching_patch(self) -> None:
        points = np.array([(x, y) for x in range(20, 45, 4) for y in range(20, 45, 4)], np.float32)
        result = alignment.estimate_alignment(alignment.MatchResult(points, points), (400, 500), (400, 500))
        self.assertFalse(result.accepted)
        self.assertTrue(any("cover too little" in reason for reason in result.reasons))

    def test_mask_distinguishes_black_content_from_missing_pixels(self) -> None:
        # A ten-pixel right translation leaves a missing strip on the left.
        H = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], np.float64)
        result = alignment.AlignmentResult(H, np.ones(12, bool), True, [])
        aligned, mask = alignment.warp_alignment(np.zeros((50, 60, 3), np.uint8), result, (50, 60))
        self.assertEqual(int(aligned.sum()), 0)
        self.assertTrue(np.all(mask[:, :10] == 0))
        self.assertTrue(np.all(mask[:, 10:] == 255))

    def test_full_bilinear_support_excludes_partially_padded_edges(self) -> None:
        # Subpixel translation mixes white source pixels with the zero constant border.
        white = np.full((18, 20, 3), 255, np.uint8)
        white[8:12, 8:12] = 0
        H = np.array([[1, 0, 0.4], [0, 1, 0.3], [0, 0, 1]], dtype=np.float64)
        result = alignment.AlignmentResult(H, np.ones(12, bool), True, [])
        warped, nearest = alignment.warp_alignment(white, result, white.shape)
        same_warp, full = alignment.warp_alignment(white, result, white.shape, full_support=True)
        np.testing.assert_array_equal(same_warp, warped)
        self.assertTrue(0 < warped[5, 0, 0] < 255)
        self.assertEqual(nearest[5, 0], 255)
        self.assertTrue(np.all(full[:, 0] == 0) and np.all(full[0, :] == 0))
        self.assertTrue(np.all(full[1:, 1:] == 255))
        self.assertEqual(warped[10, 10, 0], 0)  # A fully supported black pixel stays valid.
        result.H = np.eye(3)
        _, unchanged = alignment.warp_alignment(white, result, white.shape, full_support=True)
        self.assertTrue(np.all(unchanged == 255))  # Zero-weight neighbors at the border are harmless.

    def test_projective_horizon_is_rejected(self) -> None:
        points = np.array([[1, 1], [2, 2]], np.float32)
        horizon = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1]], np.float64)
        perspective = np.array([[1, 0.1, 5], [-0.02, 1, 8], [0.002, 0.01, 1]])
        expected = alignment.project_points(points, perspective)
        for scale in (1, -3, 1e-40, 1e40):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "horizon"):
                    alignment.project_points(points, scale * horizon)
                np.testing.assert_array_equal(alignment.project_points(points, scale * perspective), expected)

    def test_ecc_keeps_the_moving_to_reference_convention(self) -> None:
        reference = textured_image()
        reference_to_moving = np.array([[1, 0, 7], [0, 1, 4], [0, 0, 1]], np.float64)
        moving = cv2.warpPerspective(reference, reference_to_moving, (400, 320))
        initial = np.linalg.inv(reference_to_moving)
        initial[0, 2] += 0.5
        refined, correlation = alignment.refine_ecc(reference, moving, initial)
        expected = alignment.project_points(alignment.image_corners(reference.shape), np.linalg.inv(reference_to_moving))
        actual = alignment.project_points(alignment.image_corners(reference.shape), refined)
        np.testing.assert_allclose(actual, expected, atol=0.5)
        self.assertGreater(correlation, 0.95)

    def test_ecc_equivalent_projective_scales_give_the_same_refinement(self) -> None:
        reference = textured_image()
        forward = np.array([[0.96, -0.02, 16], [0.03, 0.95, 8], [0.00022, -0.0001, 1]])
        moving = cv2.warpPerspective(reference, forward, (400, 320))
        initial = np.linalg.inv(forward)
        initial[0, 2] += 0.7
        initial[1, 2] -= 0.5
        corners = alignment.image_corners(reference.shape)
        expected = alignment.project_points(corners, np.linalg.inv(forward))
        baseline, score = alignment.refine_ecc(reference, moving, initial)
        baseline_points = alignment.project_points(corners, baseline)
        np.testing.assert_allclose(baseline_points, expected, atol=0.5)
        # Positive, negative and tiny/large scales all describe the same starting warp.
        for scale in (8.0, -4.0, 1e-40, 1e40):
            with self.subTest(scale=scale):
                refined, correlation = alignment.refine_ecc(reference, moving, initial * scale)
                np.testing.assert_allclose(alignment.project_points(corners, refined), baseline_points, atol=1e-4)
                self.assertAlmostEqual(correlation, score, places=6)

    def test_ecc_rejects_an_initial_inverse_with_zero_bottom_right(self) -> None:
        # This invertible map's inverse cannot be parameterized in OpenCV ECC's h22=1 chart.
        reference = textured_image()
        swap_x_w = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.float64)
        with self.assertRaisesRegex(ValueError, "cannot normalize"):
            alignment.refine_ecc(reference, reference, swap_x_w)

    def test_validates_correspondence_and_configuration_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts differ"):
            alignment.MatchResult(np.zeros((3, 2)), np.zeros((4, 2)))
        with self.assertRaisesRegex(ValueError, "finite"):
            alignment.MatchResult([[np.nan, 0]], [[0, 0]])
        with self.assertRaises(ValueError):
            alignment.AlignmentConfig(reprojection_threshold=0)
        with self.assertRaises(ValueError):
            alignment.AlignmentConfig(min_inlier_ratio=1.1)


class EntryPointTests(unittest.TestCase):
    def test_sift_returns_original_pixel_centers_after_fractional_resize(self) -> None:
        image = textured_image(shape=(479, 733))
        matcher = alignment.FeatureMatcher("sift", resize=181)
        scale = 181 / 733  # The scaled short side is 118.28 pixels, rounded for storage.
        working = cv2.resize(image, None, fx=scale, fy=scale)
        self.assertNotEqual(working.shape[0] / image.shape[0], scale)
        direct = alignment.FeatureMatcher("sift", resize=None).extract(working)
        features = matcher.extract(image)
        self.assertGreater(len(direct.keypoints), 8)
        np.testing.assert_array_equal(direct.descriptors, features.descriptors)
        # Resizing linear x/y ramps provides independent source-coordinate ground truth.
        y_grid, x_grid = np.indices(image.shape[:2], dtype=np.float32)
        expected = []
        for grid in (x_grid, y_grid):
            ramp = cv2.resize(grid, None, fx=scale, fy=scale)
            expected.append([cv2.getRectSubPix(ramp, (1, 1), tuple(pt))[0, 0] for pt in direct.keypoints])
        np.testing.assert_allclose(features.keypoints, np.column_stack(expected), atol=2e-4, rtol=0)

    def test_blank_images_produce_a_clear_rejection(self) -> None:
        image = np.zeros((120, 160, 3), np.uint8)
        matcher = alignment.FeatureMatcher("sift")
        matches = matcher.match_images(image, image)
        self.assertEqual(len(matches.reference_points), 0)
        result = alignment.estimate_alignment(matches, image.shape, image.shape)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reasons, ["Fewer than four correspondences."])

    def test_cli_from_unrelated_directory_and_exact_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, output_dir = root / "test.png", root / "outputs"
            self.assertTrue(cv2.imwrite(str(image_path), textured_image()))
            process = subprocess.run(
                [sys.executable, str(PROJECT / "robust_image_alignment.py"),
                 "--reference", str(image_path), "--moving", str(image_path),
                 "--method", "sift", "--output-dir", str(output_dir), "--no-display", "--validate"],
                cwd=root, capture_output=True, text=True, check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn("VALIDATION PASSED", process.stdout)
            self.assertEqual({path.name for path in output_dir.iterdir()}, {
                "matches.jpg", "aligned.png", "valid_mask.png", "overlay.jpg", "reference_boundary.jpg", "report.json",
            })
            report = json.loads((output_dir / "report.json").read_text())
            self.assertTrue(report["accepted"])
            self.assertEqual(report["homography_direction"], "moving_to_reference")
            for path in output_dir.iterdir():
                if path.suffix != ".json":
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    self.assertIsNotNone(image, path)
                    self.assertGreater(image.size, 0)

    def test_cli_missing_input_and_invalid_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.png")
            process = subprocess.run(
                [sys.executable, str(PROJECT / "robust_image_alignment.py"), "--reference", missing,
                 "--moving", missing, "--method", "sift"], cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 2)
            self.assertIn("Cannot read image", process.stderr)
            invalid = subprocess.run(
                [sys.executable, str(PROJECT / "robust_image_alignment.py"), "--reference", missing,
                 "--moving", missing, "--method", "unknown"], cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("invalid choice", invalid.stderr)

    def test_rejected_run_removes_only_owned_accepted_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive, negative, output = root / "positive.png", root / "negative.png", root / "outputs"
            self.assertTrue(cv2.imwrite(str(positive), textured_image()))
            self.assertTrue(cv2.imwrite(str(negative), np.zeros((320, 400, 3), np.uint8)))
            command = [sys.executable, str(PROJECT / "robust_image_alignment.py"),
                       "--reference", str(positive), "--method", "sift", "--output-dir", str(output),
                       "--no-display", "--validate"]
            accepted = subprocess.run(command + ["--moving", str(positive)], cwd=root,
                                      capture_output=True, text=True, check=False)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            owned_outputs = {"aligned.png", "valid_mask.png", "overlay.jpg", "reference_boundary.jpg"}
            self.assertTrue(all((output / name).is_file() for name in owned_outputs))
            # Unrelated files and directories must survive the narrowly scoped cleanup.
            (output / "notes.txt").write_text("Preserve this reader's note.")
            (output / "unrelated").mkdir()
            (output / "unrelated" / "aligned.png").write_bytes(b"unrelated nested payload")
            rejected = subprocess.run(command + ["--moving", str(negative)], cwd=root,
                                      capture_output=True, text=True, check=False)
            self.assertEqual(rejected.returncode, 3, rejected.stderr)
            self.assertIn("VALIDATION FAILED", rejected.stderr)
            self.assertFalse(json.loads((output / "report.json").read_text())["accepted"])
            self.assertTrue(all(not (output / name).exists() for name in owned_outputs))
            self.assertEqual((output / "notes.txt").read_text(), "Preserve this reader's note.")
            self.assertEqual((output / "unrelated" / "aligned.png").read_bytes(), b"unrelated nested payload")
            self.assertEqual({path.name for path in output.iterdir()}, {
                "report.json", "matches.jpg", "notes.txt", "unrelated",
            })


if __name__ == "__main__":
    unittest.main()

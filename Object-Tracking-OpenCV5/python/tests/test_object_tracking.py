"""Regression tests for the unified tracking example.

The tests exercise the real command-line entry point in a subprocess, from a
temporary working directory, exactly as the OpenCV 5 migration workflow
requires. Missing external model files produce explicit skips; missing tracker
APIs, corrupt models, and runtime construction errors remain real failures.

Run with:
    python3 -m unittest discover -s python/tests -v
"""

# json parses the metrics files the application writes.
import json
# pathlib locates the application relative to this test file.
from pathlib import Path
# subprocess runs the CLI exactly as a user would.
import subprocess
# sys supplies the interpreter path so the tests honor the active venv.
import sys
# tempfile provides isolated working and output directories per test.
import tempfile
# unittest is the framework mandated by the project conventions.
import unittest

# Import the application module for constants and synthetic-video generation;
# every tracker behavior test still goes through the real subprocess CLI.
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
import object_tracking  # noqa: E402  (path setup must precede the import)

# The application script and shared models directory under test.
APP = APP_DIR / "object_tracking.py"
MODELS_DIR = APP_DIR.parent / "models"

# Only absent external model assets are an expected reason to skip a DNN
# tracker. An installed-but-broken API or a corrupt model must fail loudly.
MODEL_FILES = {
    "dasiamrpn": (
        "dasiamrpn_model.onnx",
        "dasiamrpn_kernel_cls1.onnx",
        "dasiamrpn_kernel_r1.onnx",
    ),
    "nanotrack": (
        "nanotrack_backbone_sim.onnx",
        "nanotrack_head_sim.onnx",
    ),
    "vittrack": ("object_tracking_vittrack_2023sep.onnx",),
}


def run_cli(arguments, cwd):
    """Run the application CLI in a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, str(APP), *arguments],
        cwd=cwd, capture_output=True, text=True, timeout=600, check=False)


def read_video(path):
    """Return frame count, geometry, and FPS; fail on unreadable output."""
    capture = object_tracking.cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"OpenCV cannot open {path}")
    fps = float(capture.get(object_tracking.cv2.CAP_PROP_FPS))
    frames = 0
    geometry = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        current_geometry = (frame.shape[1], frame.shape[0])
        if geometry is None:
            geometry = current_geometry
        elif geometry != current_geometry:
            raise AssertionError(f"video geometry changed inside {path}")
    capture.release()
    if frames == 0:
        raise AssertionError(f"OpenCV decoded no frames from {path}")
    return frames, geometry, fps


class ValidationPerTracker(unittest.TestCase):
    """Every tracker with downloaded model assets must pass validation."""

    def _validate_one(self, name):
        missing_models = [
            filename for filename in MODEL_FILES.get(name, ())
            if not (MODELS_DIR / filename).is_file()
        ]
        if missing_models:
            self.skipTest(
                f"{name} model files not downloaded: {', '.join(missing_models)}"
            )
        # A temporary cwd catches hidden current-directory assumptions, and a
        # temporary output dir keeps every artifact inspectable and isolated.
        with tempfile.TemporaryDirectory() as workdir:
            output_dir = Path(workdir) / "out"
            result = run_cli(
                ["--tracker", name, "--validate", "--no-display",
                 "--output-dir", str(output_dir)],
                cwd=workdir)
            # The run must succeed and print the explicit success marker.
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("VALIDATION PASSED", result.stdout)
            # No missing, stale, or unexpected artifacts may hide in the output.
            clip = output_dir / "synthetic_clip.avi"
            video = output_dir / f"tracked_{name}.avi"
            metrics_file = output_dir / f"metrics_{name}.json"
            expected = {clip, video, metrics_file}
            self.assertEqual(set(output_dir.iterdir()), expected)
            for artifact in expected:
                self.assertGreater(artifact.stat().st_size, 0)
            # Both generated videos must decode completely at the documented
            # 640x360 geometry, with one output frame per processed input frame.
            for artifact in (clip, video):
                frames, geometry, fps = read_video(artifact)
                self.assertEqual(frames, object_tracking.SYNTH_FRAMES)
                self.assertEqual(geometry, object_tracking.SYNTH_SIZE)
                self.assertAlmostEqual(fps, 30.0, places=1)
            # The stored metrics must repeat the semantics the marker claims.
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            self.assertEqual(
                set(metrics),
                {
                    "tracker", "opencv_version", "frames", "lost_frames",
                    "mean_fps", "mean_iou", "success_rate",
                },
            )
            self.assertEqual(metrics["tracker"], name)
            self.assertEqual(
                metrics["opencv_version"], object_tracking.cv2.__version__
            )
            self.assertEqual(metrics["frames"], object_tracking.SYNTH_FRAMES)
            self.assertEqual(metrics["lost_frames"], 0)
            self.assertGreaterEqual(
                metrics["mean_iou"], object_tracking.VALIDATE_MEAN_IOU)
            self.assertGreaterEqual(
                metrics["success_rate"], object_tracking.VALIDATE_SUCCESS_RATE)


# Generate one test method per tracker so results report individually.
def _make_test(name):
    def test(self):
        self._validate_one(name)
    return test


for _name in object_tracking.TRACKER_BUILDERS:
    setattr(ValidationPerTracker, f"test_validate_{_name}", _make_test(_name))


class CliBehavior(unittest.TestCase):
    """Error handling and informational modes of the CLI."""

    def test_list_trackers_reports_all_names(self):
        # --list-trackers must mention every registry entry and the version.
        with tempfile.TemporaryDirectory() as workdir:
            result = run_cli(["--list-trackers"], cwd=workdir)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("OpenCV version", result.stdout)
            for name in object_tracking.TRACKER_BUILDERS:
                self.assertIn(name, result.stdout)

    def test_missing_input_fails_cleanly(self):
        # A nonexistent input file must produce a clean nonzero exit and a
        # readable message, never a traceback.
        with tempfile.TemporaryDirectory() as workdir:
            result = run_cli(
                ["--tracker", "mil", "--input", "no_such_file.mp4",
                 "--bbox", "10,10,40,40", "--no-display"],
                cwd=workdir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Cannot open input", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_unknown_tracker_rejected_by_argparse(self):
        # argparse rejects unknown tracker names with its usual exit code 2.
        with tempfile.TemporaryDirectory() as workdir:
            result = run_cli(["--tracker", "goturn"], cwd=workdir)
            self.assertEqual(result.returncode, 2)

    def test_headless_requires_bbox(self):
        # Headless mode cannot pop the ROI selector, so it must demand --bbox.
        with tempfile.TemporaryDirectory() as workdir:
            clip = Path(workdir) / "clip.avi"
            object_tracking.make_synthetic_video(clip)
            result = run_cli(
                ["--tracker", "mil", "--input", str(clip), "--no-display"],
                cwd=workdir)
            self.assertEqual(result.returncode, 1)
            self.assertIn("--bbox is required", result.stderr)

    def test_malformed_bbox_fails_without_traceback(self):
        # A partial integer parse must never reach an OpenCV assertion.
        with tempfile.TemporaryDirectory() as workdir:
            clip = Path(workdir) / "clip.avi"
            object_tracking.make_synthetic_video(clip)
            result = run_cli(
                [
                    "--tracker", "mil", "--input", str(clip),
                    "--bbox", "1,2,nope,4", "--no-display",
                ],
                cwd=workdir,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Cannot parse --bbox", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_out_of_frame_bbox_fails_cleanly(self):
        # Geometry is checked against the actual first frame, not just syntax.
        with tempfile.TemporaryDirectory() as workdir:
            clip = Path(workdir) / "clip.avi"
            object_tracking.make_synthetic_video(clip)
            result = run_cli(
                [
                    "--tracker", "mil", "--input", str(clip),
                    "--bbox", "620,340,64,64", "--no-display",
                ],
                cwd=workdir,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("lie fully inside", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_normal_mode_preserves_frame_count(self):
        # Exercise the non-validation path and the one-frame edge case that
        # previously produced an empty annotated AVI.
        with tempfile.TemporaryDirectory() as workdir:
            clip = Path(workdir) / "clip.avi"
            output_dir = Path(workdir) / "out"
            # Use an integer rate because some minimal macOS backends quantize
            # fractional AVI rates even though the application preserves the
            # rate that VideoCapture reports.
            source_fps = 12.0
            boxes = object_tracking.make_synthetic_video(clip, fps=source_fps)
            bbox = ",".join(str(value) for value in boxes[0])
            result = run_cli(
                [
                    "--tracker", "mil", "--input", str(clip),
                    "--bbox", bbox, "--max-frames", "1", "--no-display",
                    "--output-dir", str(output_dir),
                ],
                cwd=workdir,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"tracked_mil.avi", "metrics_mil.json"},
            )
            frames, geometry, output_fps = read_video(
                output_dir / "tracked_mil.avi"
            )
            self.assertEqual(frames, 1)
            self.assertEqual(geometry, object_tracking.SYNTH_SIZE)
            self.assertAlmostEqual(output_fps, source_fps, places=1)
            metrics = json.loads(
                (output_dir / "metrics_mil.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["frames"], 1)

    def test_truncated_validation_fails(self):
        # A high-IoU prefix is not the promised 80-frame regression and must
        # never produce the validation success marker or a zero exit.
        with tempfile.TemporaryDirectory() as workdir:
            output_dir = Path(workdir) / "out"
            result = run_cli(
                [
                    "--tracker", "mil", "--validate", "--no-display",
                    "--max-frames", "2", "--output-dir", str(output_dir),
                ],
                cwd=workdir,
            )
            self.assertEqual(result.returncode, 1, msg=result.stderr)
            self.assertIn("VALIDATION FAILED", result.stdout)
            self.assertNotIn("VALIDATION PASSED", result.stdout)
            metrics = json.loads(
                (output_dir / "metrics_mil.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["frames"], 2)

    def test_output_write_failure_is_clean(self):
        # An existing file cannot serve as an output directory; report that
        # filesystem error without exposing a Python traceback.
        with tempfile.TemporaryDirectory() as workdir:
            clip = Path(workdir) / "clip.avi"
            not_a_directory = Path(workdir) / "occupied"
            object_tracking.make_synthetic_video(clip)
            not_a_directory.write_text("file", encoding="utf-8")
            result = run_cli(
                [
                    "--tracker", "mil", "--input", str(clip),
                    "--bbox", "20,148,64,64", "--no-display",
                    "--output-dir", str(not_a_directory),
                ],
                cwd=workdir,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("occupied", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_negative_max_frames_is_rejected(self):
        # argparse-style errors use exit code 2 and never run a tracker.
        with tempfile.TemporaryDirectory() as workdir:
            result = run_cli(["--max-frames", "-1"], cwd=workdir)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--max-frames must be zero or", result.stderr)


class SyntheticVideoProperties(unittest.TestCase):
    """The validation clip itself must be deterministic and well-formed."""

    def test_ground_truth_boxes_stay_inside_frame(self):
        with tempfile.TemporaryDirectory() as workdir:
            clip = Path(workdir) / "clip.avi"
            boxes = object_tracking.make_synthetic_video(clip)
            self.assertEqual(len(boxes), object_tracking.SYNTH_FRAMES)
            width, height = object_tracking.SYNTH_SIZE
            for x, y, w, h in boxes:
                # Every box must lie fully inside the frame bounds.
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + w, width)
                self.assertLessEqual(y + h, height)

    def test_clip_is_reproducible(self):
        # Two generations must produce byte-identical files (fixed seed).
        with tempfile.TemporaryDirectory() as workdir:
            clip_a = Path(workdir) / "a.avi"
            clip_b = Path(workdir) / "b.avi"
            object_tracking.make_synthetic_video(clip_a)
            object_tracking.make_synthetic_video(clip_b)
            self.assertEqual(clip_a.read_bytes(), clip_b.read_bytes())


if __name__ == "__main__":
    unittest.main()

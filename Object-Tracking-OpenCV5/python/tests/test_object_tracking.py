"""Regression tests for the unified tracking example.

The tests exercise the real command-line entry point in a subprocess, from a
temporary working directory, exactly as the OpenCV 5 migration workflow
requires. Trackers that are unavailable in the current build (contrib not
installed) or whose model files have not been downloaded are skipped with an
explicit reason rather than silently passing.

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

# Import the application module directly only to read its registry; every
# behavioral test still goes through the subprocess CLI.
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
import object_tracking  # noqa: E402  (path setup must precede the import)

# The application script and shared models directory under test.
APP = APP_DIR / "object_tracking.py"
MODELS_DIR = APP_DIR.parent / "models"


def available_trackers():
    """Return the tracker names that can actually be built right now."""
    names = []
    for name, builder in object_tracking.TRACKER_BUILDERS.items():
        try:
            if builder(MODELS_DIR) is not None:
                names.append(name)
        except Exception:  # noqa: BLE001 - any failure means unavailable
            pass
    return names


AVAILABLE = available_trackers()


def run_cli(arguments, cwd):
    """Run the application CLI in a subprocess and capture its output."""
    return subprocess.run(
        [sys.executable, str(APP), *arguments],
        cwd=cwd, capture_output=True, text=True, timeout=600, check=False)


class ValidationPerTracker(unittest.TestCase):
    """Every available tracker must pass the synthetic-clip validation."""

    def _validate_one(self, name):
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
            # The exact expected artifacts must exist and be non-empty.
            clip = output_dir / "synthetic_clip.avi"
            video = output_dir / f"tracked_{name}.avi"
            metrics_file = output_dir / f"metrics_{name}.json"
            for artifact in (clip, video, metrics_file):
                self.assertTrue(artifact.exists(), msg=f"missing {artifact}")
                self.assertGreater(artifact.stat().st_size, 0)
            # The stored metrics must repeat the thresholds the marker claims.
            metrics = json.loads(metrics_file.read_text())
            self.assertGreaterEqual(
                metrics["mean_iou"], object_tracking.VALIDATE_MEAN_IOU)
            self.assertGreaterEqual(
                metrics["success_rate"], object_tracking.VALIDATE_SUCCESS_RATE)


# Generate one test method per tracker so results report individually.
def _make_test(name):
    def test(self):
        if name not in AVAILABLE:
            self.skipTest(f"{name} unavailable in this build/model setup")
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

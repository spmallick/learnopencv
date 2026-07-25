"""Regression tests for the real Delaunay Python command-line example."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2


# Resolve the project independently of the directory from which unittest runs.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Exercise the tracked script instead of duplicating its implementation in tests.
SCRIPT_PATH = PROJECT_DIR / "delaunay.py"


def load_example_module():
    """Load the example module from its tracked path for focused helper tests."""

    # Create an import specification without requiring the folder to be a package.
    specification = importlib.util.spec_from_file_location(
        "delaunay_example",
        SCRIPT_PATH,
    )
    # A missing loader indicates a broken or unreadable script path.
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    # Construct the module object that the loader will populate.
    module = importlib.util.module_from_spec(specification)
    # Dataclass evaluation expects the in-progress module to be discoverable.
    sys.modules[specification.name] = module
    # Execute definitions without invoking the script's __main__ block.
    specification.loader.exec_module(module)
    # Return the real implementation for helper-level boundary checks.
    return module


class DelaunayCliTests(unittest.TestCase):
    """Validate rendering, error handling, and working-directory independence."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import the tracked implementation once for focused helper checks."""

        # Keep the module on the test class to avoid repeated import side effects.
        cls.example = load_example_module()

    def run_cli(
        self,
        *arguments: str,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Run the real CLI with the current exact OpenCV installation."""

        # sys.executable ensures the subprocess uses the version under test.
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *arguments],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_bundled_cli_from_unrelated_directory(self) -> None:
        """The validated example should not depend on the caller's directory."""

        # Temporary directories keep generated files out of the repository.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Separate the unrelated working directory from the artifact directory.
            working_dir = root / "working"
            output_dir = root / "output"
            # Create only the working directory; the program must create output itself.
            working_dir.mkdir()
            # Run the complete headless validation path.
            result = self.run_cli(
                "--no-display",
                "--no-animation",
                "--validate",
                "--output-dir",
                str(output_dir),
                cwd=working_dir,
            )

            # Surface captured output if the real entry point fails.
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + result.stderr,
            )
            # The marker appears only after geometry and output checks pass.
            self.assertIn("DELAUNAY_VALIDATION_OK", result.stdout)
            # The CLI reports the exact library used by this test environment.
            self.assertIn(f"OpenCV version: {cv2.__version__}", result.stdout)
            # Only the two documented artifacts should be generated.
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {"delaunay.png", "voronoi.png"},
            )

            # Decode each artifact to test the files rather than in-memory arrays.
            for filename in ("delaunay.png", "voronoi.png"):
                image = cv2.imread(
                    str(output_dir / filename),
                    cv2.IMREAD_COLOR,
                )
                # A missing or corrupt PNG would decode as None.
                self.assertIsNotNone(image)
                # Bundled outputs must preserve the source image dimensions.
                self.assertEqual(image.shape, (697, 512, 3))

    def test_repeated_runs_are_deterministic(self) -> None:
        """The shared palette should remove the old random Voronoi output."""

        # Use one temporary root with separate outputs for two complete executions.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Create an unrelated current directory once for both executions.
            working_dir = root / "working"
            working_dir.mkdir()
            # Run the real CLI twice with identical inputs and independent destinations.
            for run_number in (1, 2):
                result = self.run_cli(
                    "--no-display",
                    "--no-animation",
                    "--validate",
                    "--output-dir",
                    str(root / f"output-{run_number}"),
                    cwd=working_dir,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=result.stdout + result.stderr,
                )

            # Lossless outputs should repeat byte-for-byte in one environment.
            for filename in ("delaunay.png", "voronoi.png"):
                first = (root / "output-1" / filename).read_bytes()
                second = (root / "output-2" / filename).read_bytes()
                self.assertEqual(first, second)

    def test_missing_image_fails_cleanly(self) -> None:
        """A missing image should return a controlled error without a traceback."""

        # Keep the nonexistent path and unused output destination isolated.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Run from the temporary directory to exercise absolute default assets.
            result = self.run_cli(
                "--image",
                str(root / "missing.jpg"),
                "--output-dir",
                str(root / "output"),
                "--no-display",
                cwd=root,
            )
            # Runtime input errors use the documented nonzero status.
            self.assertEqual(result.returncode, 2)
            # The message identifies the exact unusable input.
            self.assertIn("Could not read input image", result.stderr)
            # Expected failures remain concise for tutorial users.
            self.assertNotIn("Traceback", result.stderr)

    def test_invalid_points_fail_cleanly(self) -> None:
        """Invalid point records should be rejected before Subdiv2D insertion."""

        # Build two focused invalid files in an isolated directory.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # A third column violates the two-coordinate file contract.
            malformed = root / "malformed.txt"
            malformed.write_text("10 20 30\n", encoding="utf-8")
            # x == width is outside OpenCV's half-open 512-pixel rectangle.
            outside = root / "outside.txt"
            outside.write_text(
                "10 20\n20 30\n512 40\n",
                encoding="utf-8",
            )
            # Fractional coordinates are outside this raster tutorial's contract.
            fractional = root / "fractional.txt"
            fractional.write_text(
                "10 20\n20 30\n40.5 50\n",
                encoding="utf-8",
            )

            # Exercise parse, bounds, and integer-contract failures through the CLI.
            cases = (
                (malformed, "expected two coordinates"),
                (outside, "is outside"),
                (fractional, "coordinates must be integers"),
            )
            for points_path, message in cases:
                result = self.run_cli(
                    "--points",
                    str(points_path),
                    "--output-dir",
                    str(root / points_path.stem),
                    "--no-display",
                    cwd=root,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_outputs_cannot_overwrite_inputs(self) -> None:
        """Both documented output names must be protected from input collision."""

        # Check the image and point-file collision paths independently.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("image", "delaunay.png"),
                ("points", "voronoi.png"),
            )
            for input_kind, output_name in cases:
                with self.subTest(input_kind=input_kind):
                    # Give each case a fresh output directory.
                    case_dir = root / input_kind
                    case_dir.mkdir()
                    # Start with the normal bundled inputs.
                    image_path = PROJECT_DIR / "obama.jpg"
                    points_path = PROJECT_DIR / "obama.txt"
                    # Copy one input onto the output filename being tested.
                    colliding_path = case_dir / output_name
                    if input_kind == "image":
                        shutil.copyfile(image_path, colliding_path)
                        image_path = colliding_path
                    else:
                        shutil.copyfile(points_path, colliding_path)
                        points_path = colliding_path
                    # Preserve the exact source bytes for the no-overwrite assertion.
                    before = colliding_path.read_bytes()

                    # The safety check must run before either output is written.
                    result = self.run_cli(
                        "--image",
                        str(image_path),
                        "--points",
                        str(points_path),
                        "--output-dir",
                        str(case_dir),
                        "--no-display",
                        cwd=root,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        "Output path would overwrite an input file",
                        result.stderr,
                    )
                    # The colliding input remains byte-for-byte unchanged.
                    self.assertEqual(colliding_path.read_bytes(), before)

            # Existing hard links use different pathnames for the same input inode.
            hardlink_dir = root / "hardlink"
            hardlink_dir.mkdir()
            source_image = hardlink_dir / "source.jpg"
            shutil.copyfile(PROJECT_DIR / "obama.jpg", source_image)
            linked_output = hardlink_dir / "delaunay.png"
            os.link(source_image, linked_output)
            before = source_image.read_bytes()
            result = self.run_cli(
                "--image",
                str(source_image),
                "--points",
                str(PROJECT_DIR / "obama.txt"),
                "--output-dir",
                str(hardlink_dir),
                "--no-display",
                cwd=root,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn(
                "Output path would overwrite an input file",
                result.stderr,
            )
            # Neither pathname may expose mutated source bytes after rejection.
            self.assertEqual(source_image.read_bytes(), before)
            self.assertEqual(linked_output.read_bytes(), before)

    def test_rectangle_uses_half_open_boundaries(self) -> None:
        """The Python helper should match cv::Rect for nonzero origins."""

        # Left/top is included.
        self.assertTrue(self.example.rect_contains((10, 20, 5, 7), (10, 20)))
        # The last representable interior coordinate is included.
        self.assertTrue(
            self.example.rect_contains((10, 20, 5, 7), (14.999, 26.999))
        )
        # Right and bottom boundaries are excluded.
        self.assertFalse(self.example.rect_contains((10, 20, 5, 7), (15, 25)))
        self.assertFalse(self.example.rect_contains((10, 20, 5, 7), (12, 27)))

    def test_animation_requires_display(self) -> None:
        """Invisible animation should be rejected as a command-line error."""

        # Use an unrelated directory even though parsing should stop before I/O.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The mutually meaningful combination is --display --animate.
            result = self.run_cli(
                "--animate",
                "--no-display",
                cwd=root,
            )
            # argparse exits with two for an invalid command-line combination.
            self.assertEqual(result.returncode, 2)
            self.assertIn("--animate requires --display", result.stderr)


if __name__ == "__main__":
    # Support direct execution as well as unittest discovery.
    unittest.main()

"""Tests for the Snake game engine and its real command-line entry point."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "snake.py"

SPEC = importlib.util.spec_from_file_location("snake_example", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SNAKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SNAKE
SPEC.loader.exec_module(SNAKE)


class SnakeGameTest(unittest.TestCase):
    """Validate game behavior without relying on a desktop window."""

    def test_engine_regression_contract(self) -> None:
        SNAKE.validate_engine()

    def test_headless_cli_from_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_path = temporary_path / "final-board.png"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--no-display",
                    "--validate",
                    "--board-size",
                    "9",
                    "--cell-size",
                    "8",
                    "--max-steps",
                    "12",
                    "--output",
                    str(output_path),
                ],
                cwd=temporary_path,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )
            self.assertIn("VALIDATION PASSED", completed.stdout)
            image = cv2.imread(str(output_path))
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, (72, 72, 3))

    def test_reverse_direction_is_ignored_for_long_snake(self) -> None:
        game = SNAKE.SnakeGame(board_size=9)
        game.snake = [
            SNAKE.Position(4, 4),
            SNAKE.Position(3, 4),
        ]
        game.direction = "right"
        game.set_direction("left")
        self.assertEqual(game.direction, "right")


if __name__ == "__main__":
    unittest.main()

"""Play a small Snake game rendered with OpenCV 4.14 or OpenCV 5."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import cv2
import numpy as np


# Directions use (x, y) deltas so the game logic is independent of the UI.
DIRECTION_VECTORS = {
    "right": (1, 0),
    "down": (0, 1),
    "left": (-1, 0),
    "up": (0, -1),
}
OPPOSITE_DIRECTION = {
    "right": "left",
    "left": "right",
    "up": "down",
    "down": "up",
}


@dataclass(frozen=True)
class Position:
    """One cell on the square game board."""

    x: int
    y: int


class SnakeGame:
    """Deterministic game state that can be tested without a display."""

    def __init__(
        self,
        *,
        board_size: int = 45,
        growth_per_apple: int = 3,
        seed: int = 7,
    ) -> None:
        if board_size < 5:
            raise ValueError("board_size must be at least 5")
        if growth_per_apple < 1:
            raise ValueError("growth_per_apple must be positive")

        self.board_size = board_size
        self.growth_per_apple = growth_per_apple
        self.random = random.Random(seed)

        center = board_size // 2
        self.snake = [Position(center, center)]
        self.direction = "right"
        self.pending_growth = 0
        self.score = 0
        self.game_over = False
        self.apple = self._new_apple()

    def _new_apple(self) -> Position:
        """Choose an empty board cell using the instance's seeded generator."""

        occupied = set(self.snake)
        available = [
            Position(x, y)
            for y in range(self.board_size)
            for x in range(self.board_size)
            if Position(x, y) not in occupied
        ]
        if not available:
            raise RuntimeError("The snake fills the board; no apple cell remains")
        return self.random.choice(available)

    def set_direction(self, direction: str) -> None:
        """Accept a direction unless it would reverse a multi-cell snake."""

        if direction not in DIRECTION_VECTORS:
            raise ValueError(f"Unknown direction: {direction}")
        if (
            len(self.snake) > 1
            and direction == OPPOSITE_DIRECTION[self.direction]
        ):
            return
        self.direction = direction

    def step(self) -> bool:
        """Advance one tick and return False after a wall or self collision."""

        if self.game_over:
            return False

        delta_x, delta_y = DIRECTION_VECTORS[self.direction]
        head = self.snake[0]
        new_head = Position(head.x + delta_x, head.y + delta_y)

        outside_board = not (
            0 <= new_head.x < self.board_size
            and 0 <= new_head.y < self.board_size
        )
        if outside_board:
            self.game_over = True
            return False

        # The tail moves away when no growth is pending, so that cell is safe.
        collision_body = (
            self.snake if self.pending_growth > 0 else self.snake[:-1]
        )
        if new_head in collision_body:
            self.game_over = True
            return False

        self.snake.insert(0, new_head)
        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            self.snake.pop()

        if new_head == self.apple:
            self.score += 1
            self.pending_growth += self.growth_per_apple
            self.apple = self._new_apple()
        return True

    def render(self, cell_size: int = 20) -> np.ndarray:
        """Render the current game state as an 8-bit BGR image."""

        if cell_size < 1:
            raise ValueError("cell_size must be positive")

        board = np.zeros(
            (self.board_size, self.board_size, 3),
            dtype=np.uint8,
        )
        for index, part in enumerate(self.snake):
            board[part.y, part.x] = (255, 180, 0) if index == 0 else (0, 200, 0)
        board[self.apple.y, self.apple.x] = (0, 0, 255)

        rendered = cv2.resize(
            board,
            (self.board_size * cell_size, self.board_size * cell_size),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.putText(
            rendered,
            f"Score: {self.score}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return rendered


def direction_from_key(key: int) -> str | None:
    """Map WASD and common OpenCV arrow-key codes to game directions."""

    mapping = {
        ord("d"): "right",
        ord("s"): "down",
        ord("a"): "left",
        ord("w"): "up",
        83: "right",
        84: "down",
        81: "left",
        82: "up",
        2555904: "right",
        2621440: "down",
        2424832: "left",
        2490368: "up",
    }
    return mapping.get(key)


def validate_engine() -> None:
    """Exercise movement, growth, collision, and rendering invariants."""

    game = SnakeGame(board_size=7, growth_per_apple=2, seed=3)
    game.apple = Position(game.snake[0].x + 1, game.snake[0].y)

    if not game.step() or game.score != 1 or game.pending_growth != 2:
        raise RuntimeError("Eating an apple did not update score and growth")
    if not game.step() or len(game.snake) != 2:
        raise RuntimeError("Pending growth did not retain the snake tail")
    if game.apple in game.snake:
        raise RuntimeError("A new apple overlaps the snake")

    rendered = game.render(cell_size=4)
    if rendered.shape != (28, 28, 3) or rendered.dtype != np.uint8:
        raise RuntimeError(
            f"Unexpected rendered board: shape={rendered.shape}, "
            f"dtype={rendered.dtype}"
        )

    wall_game = SnakeGame(board_size=5, seed=1)
    for _ in range(3):
        wall_game.step()
    if not wall_game.game_over:
        raise RuntimeError("Wall collision did not end the game")


def run_scripted(
    game: SnakeGame,
    directions: Iterable[str],
) -> None:
    """Apply deterministic directions for a headless demonstration."""

    for direction in directions:
        game.set_direction(direction)
        if not game.step():
            break


def parse_args() -> argparse.Namespace:
    """Parse interactive, headless, and regression-test options."""

    parser = argparse.ArgumentParser(
        description="Play Snake with an OpenCV-rendered board."
    )
    parser.add_argument("--board-size", type=int, default=45)
    parser.add_argument("--cell-size", type=int, default=20)
    parser.add_argument("--speed", type=float, default=12.0)
    parser.add_argument("--growth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run a deterministic simulation without opening a window.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Number of headless simulation steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/snake-final-board.png"),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run deterministic game-engine regression checks.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the interactive game or its deterministic headless counterpart."""

    args = parse_args()
    try:
        if args.speed <= 0:
            raise ValueError("speed must be positive")
        if args.max_steps < 0:
            raise ValueError("max_steps cannot be negative")

        if args.validate:
            validate_engine()

        game = SnakeGame(
            board_size=args.board_size,
            growth_per_apple=args.growth,
            seed=args.seed,
        )

        if args.no_display:
            scripted_pattern = ("right", "down", "left", "up")
            scripted_directions = (
                scripted_pattern[index % len(scripted_pattern)]
                for index in range(args.max_steps)
            )
            run_scripted(game, scripted_directions)
        else:
            print("Use W/A/S/D or arrow keys. Press Esc or Q to quit.")
            delay_milliseconds = max(1, round(1000 / args.speed))
            while not game.game_over:
                cv2.imshow("Snake Game", game.render(args.cell_size))
                key = cv2.waitKeyEx(delay_milliseconds)
                if key in (27, ord("q")):
                    break
                direction = direction_from_key(key)
                if direction is not None:
                    game.set_direction(direction)
                game.step()
            cv2.destroyAllWindows()

        final_board = game.render(args.cell_size)
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), final_board):
            raise OSError(f"Unable to write final board: {output_path}")

        print(f"OpenCV version: {cv2.__version__}")
        print(f"Score: {game.score}")
        print(f"Snake length: {len(game.snake)}")
        print(f"Final board: {output_path}")
        if args.validate:
            print("VALIDATION PASSED: movement, growth, collision, and render")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

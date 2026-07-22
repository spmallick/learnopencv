"""Show why raw B, G, and R channels are poor contour inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "input" / "image_1.jpg"


def show_image(title: str, image, display: bool) -> None:
    if display:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run(input_path: Path, output_dir: Path, display: bool = True) -> dict[str, int]:
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    # OpenCV stores color images in B, G, R order.
    for name, channel in zip(("blue", "green", "red"), cv2.split(image)):
        contours, _ = cv2.findContours(
            channel.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        rendered = image.copy()
        cv2.drawContours(rendered, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

        show_image(f"Contours from the {name} channel", rendered, display)
        output_path = output_dir / f"{name}_channel.jpg"
        if not cv2.imwrite(str(output_path), rendered):
            raise OSError(f"Could not write output image: {output_path}")

        counts[name] = len(contours)
        print(f"{name}: contours={len(contours)}")

    return counts


def validate(counts: dict[str, int]) -> None:
    if not all(count > 0 for count in counts.values()):
        raise RuntimeError("Every color channel should produce at least one contour")
    if not counts["blue"] < counts["green"] < counts["red"]:
        raise RuntimeError(
            "Expected the bundled image to produce blue < green < red contour counts"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--no-display", action="store_true", help="Skip GUI windows")
    parser.add_argument(
        "--validate", action="store_true", help="Validate bundled-image invariants"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.input, args.output_dir, display=not args.no_display)
    if args.validate:
        validate(result)
        print("Validation passed")

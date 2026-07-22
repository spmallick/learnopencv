"""Compare contour retrieval modes using OpenCV 4 or OpenCV 5."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "input" / "custom_colors.jpg"
MODES = {
    "list": cv2.RETR_LIST,
    "external": cv2.RETR_EXTERNAL,
    "ccomp": cv2.RETR_CCOMP,
    "tree": cv2.RETR_TREE,
}


def show_image(title: str, image, display: bool) -> None:
    """Display an image when running interactively."""
    if display:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run(input_path: Path, output_dir: Path, display: bool = True) -> dict[str, int]:
    """Run the retrieval-mode experiment and return contour counts."""
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    counts: dict[str, int] = {}
    for name, mode in MODES.items():
        contours, hierarchy = cv2.findContours(
            threshold_image.copy(), mode, cv2.CHAIN_APPROX_NONE
        )
        rendered = image.copy()
        cv2.drawContours(rendered, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

        show_image(name.upper(), rendered, display)
        output_path = output_dir / f"contours_retr_{name}.jpg"
        if not cv2.imwrite(str(output_path), rendered):
            raise OSError(f"Could not write output image: {output_path}")

        counts[name] = len(contours)
        hierarchy_shape = None if hierarchy is None else hierarchy.shape
        print(f"{name.upper()}: contours={len(contours)}, hierarchy={hierarchy_shape}")

    return counts


def validate(counts: dict[str, int]) -> None:
    """Check relationships that should hold for the bundled tutorial image."""
    if not all(count > 0 for count in counts.values()):
        raise RuntimeError("Every retrieval mode should find at least one contour")
    if counts["external"] > counts["list"]:
        raise RuntimeError("RETR_EXTERNAL cannot return more contours than RETR_LIST")
    if counts["ccomp"] != counts["tree"]:
        raise RuntimeError("The bundled image should have equal CCOMP and TREE counts")


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

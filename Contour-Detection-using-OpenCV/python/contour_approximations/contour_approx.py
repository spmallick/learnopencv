"""Compare CHAIN_APPROX_NONE and CHAIN_APPROX_SIMPLE."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


INPUT_DIR = Path(__file__).resolve().parents[2] / "input"


def read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {path}")
    return image


def show_image(title: str, image, display: bool) -> None:
    if display:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def write_image(path: Path, image) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write output image: {path}")


def find_binary_contours(image, method: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        threshold_image.copy(), cv2.RETR_TREE, method
    )
    return threshold_image, contours, hierarchy


def run(
    image1_path: Path,
    image2_path: Path,
    output_dir: Path,
    display: bool = True,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    image1 = read_image(image1_path)
    threshold_image, contours_none, _ = find_binary_contours(
        image1, cv2.CHAIN_APPROX_NONE
    )
    _, contours_simple, _ = find_binary_contours(image1, cv2.CHAIN_APPROX_SIMPLE)

    show_image("Binary image", threshold_image, display)
    write_image(output_dir / "image_thres1.jpg", threshold_image)

    for name, contours in (("none", contours_none), ("simple", contours_simple)):
        rendered = image1.copy()
        cv2.drawContours(rendered, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        show_image(f"{name.title()} approximation", rendered, display)
        write_image(output_dir / f"contours_{name}_image1.jpg", rendered)

    image2 = read_image(image2_path)
    _, contours_image2, _ = find_binary_contours(image2, cv2.CHAIN_APPROX_SIMPLE)
    rendered = image2.copy()
    cv2.drawContours(rendered, contours_image2, -1, (0, 255, 0), 2, cv2.LINE_AA)
    show_image("SIMPLE approximation contours", rendered, display)
    write_image(output_dir / "contours_simple_image2.jpg", rendered)

    points_only = image2.copy()
    for contour in contours_image2:
        for contour_point in contour:
            x, y = contour_point[0]
            cv2.circle(points_only, (int(x), int(y)), 2, (0, 255, 0), 2, cv2.LINE_AA)
    show_image("CHAIN_APPROX_SIMPLE points", points_only, display)
    write_image(output_dir / "contour_point_simple.jpg", points_only)

    metrics = {
        "none_contours": len(contours_none),
        "none_points": sum(len(contour) for contour in contours_none),
        "simple_contours": len(contours_simple),
        "simple_points": sum(len(contour) for contour in contours_simple),
        "image2_simple_contours": len(contours_image2),
        "image2_simple_points": sum(len(contour) for contour in contours_image2),
    }
    print(", ".join(f"{name}={value}" for name, value in metrics.items()))
    return metrics


def validate(metrics: dict[str, int]) -> None:
    if metrics["none_contours"] != metrics["simple_contours"]:
        raise RuntimeError("Approximation methods should find the same contours")
    if metrics["simple_points"] >= metrics["none_points"]:
        raise RuntimeError("CHAIN_APPROX_SIMPLE should retain fewer contour points")
    if metrics["image2_simple_contours"] <= 0:
        raise RuntimeError("The second bundled image should contain contours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image1", type=Path, default=INPUT_DIR / "image_1.jpg")
    parser.add_argument("--image2", type=Path, default=INPUT_DIR / "image_2.jpg")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--no-display", action="store_true", help="Skip GUI windows")
    parser.add_argument(
        "--validate", action="store_true", help="Validate bundled-image invariants"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(
        args.image1, args.image2, args.output_dir, display=not args.no_display
    )
    if args.validate:
        validate(result)
        print("Validation passed")

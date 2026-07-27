"""OpenCV's Otsu thresholding call with reproducible, headless outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "boat.jpg"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"


def call_otsu_threshold(
    img_title: str | Path = DEFAULT_INPUT,
    is_reduce_noise: bool = False,
    output_dir: str | Path | None = None,
) -> tuple[float, np.ndarray]:
    """Apply OpenCV Otsu thresholding and optionally save plots and output."""

    image_path = Path(img_title).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read grayscale image: {image_path}")

    processed = cv2.GaussianBlur(image, (5, 5), 0) if is_reduce_noise else image
    threshold, binary = cv2.threshold(
        processed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots()
        axis.hist(processed.reshape(-1), bins=256, range=(0, 256))
        axis.set(xlabel="Intensity", ylabel="Pixel count")
        figure.tight_layout()
        figure.savefig(destination / "input-histogram.png", dpi=160)
        plt.close(figure)

        figure, axis = plt.subplots()
        axis.hist(binary.reshape(-1), bins=256, range=(0, 256))
        axis.set(xlabel="Intensity", ylabel="Pixel count")
        figure.tight_layout()
        figure.savefig(destination / "binary-histogram.png", dpi=160)
        plt.close(figure)

        output_path = destination / "otsu-opencv.png"
        if not cv2.imwrite(str(output_path), binary):
            raise OSError(f"Could not write output image: {output_path}")

    return threshold, binary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply OpenCV Otsu thresholding.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blur", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    threshold, _ = call_otsu_threshold(
        args.input, is_reduce_noise=args.blur, output_dir=args.output_dir
    )
    print(f"OpenCV Otsu threshold: {threshold:g}")
    print(f"Saved outputs under: {args.output_dir.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Colorize a still image with OpenCV DNN and an ONNX model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2 as cv

from colorization import DEFAULT_MODEL, colorize_frame, load_network, validate_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colorize a grayscale or desaturated image with OpenCV DNN."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "greyscaleImage.png",
        help="Input image (default: the bundled sample).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to colorization_eccv16.onnx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("colorized-image.png"),
        help="Destination image.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV preview window.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check the output dimensions, type, and predicted chroma.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv.imread(str(args.input), cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    network = load_network(args.model)
    start = time.perf_counter()
    output, chroma_score = colorize_frame(image, network)
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(args.output), output):
        raise RuntimeError(f"Could not write output image: {args.output}")
    if args.validate:
        validate_output(image, output, chroma_score)

    print(f"Saved {args.output}")
    print(f"Inference time: {elapsed:.3f} seconds")
    print(f"Mean predicted chroma: {chroma_score:.3f}")

    if not args.no_display:
        comparison = cv.hconcat([image, output])
        cv.imshow("Input | Colorized", comparison)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

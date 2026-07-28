#!/usr/bin/env python3
"""Preprocess an image with OpenCV and recognize text with Tesseract 5."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
PREPROCESSING_MODES = ("none", "gray", "otsu", "adaptive")


def preprocess_image(image: Image, mode: str = "gray") -> Image:
    """Prepare an OpenCV image for OCR without changing its geometry."""
    if image is None or image.size == 0:
        raise ValueError("preprocess_image expects a non-empty image")
    if mode not in PREPROCESSING_MODES:
        raise ValueError(
            f"unknown preprocessing mode {mode!r}; choose from "
            + ", ".join(PREPROCESSING_MODES)
        )
    if mode == "none":
        return image.copy()

    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    if mode == "gray":
        return gray
    if mode == "otsu":
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresholded = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return thresholded
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def run_tesseract(
    image: Image,
    *,
    executable: str = "tesseract",
    language: str = "eng",
    oem: int = 1,
    psm: int = 6,
) -> str:
    """Run the Tesseract CLI on an in-memory OpenCV image."""
    if image is None or image.size == 0:
        raise ValueError("run_tesseract expects a non-empty image")
    if not 0 <= oem <= 3:
        raise ValueError("oem must be in the range [0, 3]")
    if not 0 <= psm <= 13:
        raise ValueError("psm must be in the range [0, 13]")
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"Tesseract executable not found: {executable}. Install "
            "Tesseract 5 or pass --tesseract with its path."
        )

    with tempfile.TemporaryDirectory(prefix="learnopencv-ocr-") as temporary:
        image_path = Path(temporary) / "input.png"
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"OpenCV could not write temporary image: {image_path}")
        command = [
            executable,
            str(image_path),
            "stdout",
            "-l",
            language,
            "--oem",
            str(oem),
            "--psm",
            str(psm),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "unknown Tesseract error"
        raise RuntimeError(f"Tesseract failed ({completed.returncode}): {message}")
    return completed.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recognize text after optional OpenCV preprocessing."
    )
    parser.add_argument("image", type=Path, help="Input image")
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESSING_MODES,
        default="gray",
        help="Preprocessing mode (default: gray)",
    )
    parser.add_argument("--lang", default="eng", help="Tesseract language code")
    parser.add_argument(
        "--oem",
        type=int,
        choices=range(4),
        default=1,
        metavar="0..3",
        help="Tesseract OCR engine mode (default: 1, LSTM)",
    )
    parser.add_argument(
        "--psm",
        type=int,
        choices=range(14),
        default=6,
        metavar="0..13",
        help="Tesseract page-segmentation mode (default: 6)",
    )
    parser.add_argument(
        "--tesseract",
        default="tesseract",
        help="Tesseract executable name or path",
    )
    parser.add_argument("--output", type=Path, help="Optional UTF-8 text output")
    parser.add_argument(
        "--save-preprocessed", type=Path, help="Optional preprocessed image output"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: could not read input image: {args.image}")
        return 2

    try:
        prepared = preprocess_image(image, args.preprocess)
        text = run_tesseract(
            prepared,
            executable=args.tesseract,
            language=args.lang,
            oem=args.oem,
            psm=args.psm,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 2

    try:
        if args.save_preprocessed:
            args.save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save_preprocessed), prepared):
                raise OSError(
                    "OpenCV could not write preprocessed image: "
                    f"{args.save_preprocessed}"
                )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
    except (OSError, cv2.error) as error:
        print(f"error: {error}")
        return 2

    print(text, end="" if text.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

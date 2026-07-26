#!/usr/bin/env python3
"""Score an image with OpenCV's no-reference BRISQUE implementation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "brisque_model_live.yml"
DEFAULT_RANGE_PATH = PROJECT_DIR / "models" / "brisque_range_live.yml"


def _require_quality_module() -> None:
    if not hasattr(cv2, "quality") or not hasattr(
        cv2.quality, "QualityBRISQUE_compute"
    ):
        raise RuntimeError(
            "This example requires OpenCV's quality module. Install the "
            "'opencv-contrib-python' package listed in requirements.txt."
        )


def calculate_brisque_score(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    range_path: str | Path = DEFAULT_RANGE_PATH,
) -> float:
    """Return the BRISQUE score for an image (lower generally means better)."""
    _require_quality_module()

    image_file = Path(image_path)
    model_file = Path(model_path)
    range_file = Path(range_path)

    if not image_file.is_file():
        raise FileNotFoundError(f"Could not read image: {image_file}")
    if not model_file.is_file():
        raise FileNotFoundError(f"Could not read BRISQUE model: {model_file}")
    if not range_file.is_file():
        raise FileNotFoundError(f"Could not read BRISQUE range: {range_file}")

    image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_file}")

    score = cv2.quality.QualityBRISQUE_compute(
        image, str(model_file), str(range_file)
    )
    return float(score[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate a no-reference BRISQUE image-quality score."
    )
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="OpenCV BRISQUE SVM model YAML",
    )
    parser.add_argument(
        "--range",
        dest="range_path",
        type=Path,
        default=DEFAULT_RANGE_PATH,
        help="OpenCV BRISQUE feature-range YAML",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        score = calculate_brisque_score(args.image, args.model, args.range_path)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        parser.error(str(error))

    print(f"BRISQUE score: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

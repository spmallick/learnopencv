#!/usr/bin/env python3
"""Align one image to another with OpenCV's ECC optimizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ecc_utils import MOTION_MODELS, align_image, mean_absolute_error


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=SCRIPT_DIR / "images" / "image1.jpg")
    parser.add_argument("--moving", type=Path, default=SCRIPT_DIR / "images" / "image2.jpg")
    parser.add_argument(
        "--motion", choices=sorted(MOTION_MODELS), default="euclidean"
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "output" / "image2-aligned.jpg"
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = cv2.imread(str(args.template))
    moving = cv2.imread(str(args.moving))
    if template is None:
        raise FileNotFoundError(f"Could not read template: {args.template}")
    if moving is None:
        raise FileNotFoundError(f"Could not read moving image: {args.moving}")

    correlation, warp, aligned = align_image(
        template,
        moving,
        MOTION_MODELS[args.motion],
        args.iterations,
        args.epsilon,
    )
    moving_for_comparison = (
        moving
        if moving.shape[:2] == template.shape[:2]
        else cv2.resize(
            moving,
            (template.shape[1], template.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    )
    before = mean_absolute_error(template, moving_for_comparison)
    after = mean_absolute_error(template, aligned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), aligned):
        raise RuntimeError(f"Could not write output: {args.output}")

    print(f"OpenCV: {cv2.__version__}")
    print(f"Motion model: {args.motion}")
    print(f"ECC correlation: {correlation:.8f}")
    print(f"Warp matrix:\n{warp}")
    print(f"Mean absolute error: {before:.3f} -> {after:.3f}")
    print(f"Saved: {args.output}")

    if args.validate:
        if (
            not np.isfinite(warp).all()
            or not np.isfinite(correlation)
            or correlation <= 0.0
            or after >= before
        ):
            raise RuntimeError("ECC pair-alignment validation failed")
        print("Validation: PASS")

    if not args.no_display:
        cv2.imshow("Template", template)
        cv2.imshow("Moving", moving_for_comparison)
        cv2.imshow("Aligned", aligned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

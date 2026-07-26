#!/usr/bin/env python3
"""Upscale an image with OpenCV's DNN super-resolution module."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
SUPPORTED_SCALES = {
    "edsr": {2, 3, 4},
    "espcn": {2, 3, 4},
    "fsrcnn": {2, 3, 4},
    "lapsrn": {2, 4, 8},
}


def validate_model_configuration(algorithm: str, scale: int) -> None:
    """Validate an algorithm/scale pair before loading a model."""
    valid_scales = SUPPORTED_SCALES.get(algorithm)
    if valid_scales is None:
        choices = ", ".join(sorted(SUPPORTED_SCALES))
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose from: {choices}.")
    if scale not in valid_scales:
        choices = ", ".join(str(value) for value in sorted(valid_scales))
        raise ValueError(
            f"Scale x{scale} is not supported for {algorithm}. "
            f"Choose from: {choices}."
        )


def upscale_image(
    image,
    model_path: Path,
    algorithm: str = "espcn",
    scale: int = 4,
):
    """Return an image enlarged by the requested DNN super-resolution model."""
    if image is None or image.size == 0:
        raise ValueError("The input image is empty.")

    validate_model_configuration(algorithm, scale)
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run 'python download_model.py' "
            "or pass --model."
        )

    if not hasattr(cv2, "dnn_superres"):
        raise RuntimeError(
            "This OpenCV build does not include dnn_superres. "
            "Install opencv-contrib-python or build OpenCV with opencv_contrib."
        )

    super_resolver = cv2.dnn_superres.DnnSuperResImpl_create()
    super_resolver.readModel(str(model_path))
    super_resolver.setModel(algorithm, scale)
    result = super_resolver.upsample(image)

    expected_size = (image.shape[0] * scale, image.shape[1] * scale)
    if result.shape[:2] != expected_size:
        raise RuntimeError(
            "Unexpected output dimensions: "
            f"got {result.shape[1]}x{result.shape[0]}, "
            f"expected {expected_size[1]}x{expected_size[0]}."
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale an image with OpenCV DNN super resolution."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_DIR / "image.png",
        help="Input image (default: the bundled image.png).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "models" / "ESPCN_x4.pb",
        help="TensorFlow .pb model file.",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(SUPPORTED_SCALES),
        default="espcn",
        help="Model architecture (default: espcn).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Upscaling factor encoded by the model (default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "output.png",
        help="Output image path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Input image not found or unreadable: {args.input}")

        result = upscale_image(
            image=image,
            model_path=args.model,
            algorithm=args.algorithm,
            scale=args.scale,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), result):
            raise OSError(f"OpenCV could not write: {args.output}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(
        f"Upscaled {image.shape[1]}x{image.shape[0]} -> "
        f"{result.shape[1]}x{result.shape[0]} and wrote {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

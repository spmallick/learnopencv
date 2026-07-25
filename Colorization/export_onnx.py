#!/usr/bin/env python3
"""Reproduce the OpenCV colorization ONNX asset from the upstream model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export richzhang/colorization ECCV16 to ONNX."
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        required=True,
        help="Checkout of https://github.com/richzhang/colorization.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/colorization_eccv16.onnx"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    upstream = args.upstream_dir.expanduser().resolve()
    if not (upstream / "colorizers" / "eccv16.py").is_file():
        raise FileNotFoundError(
            f"Expected a richzhang/colorization checkout at {upstream}"
        )

    sys.path.insert(0, str(upstream))
    from colorizers.eccv16 import eccv16  # pylint: disable=import-outside-toplevel

    model = eccv16(pretrained=True).cpu().eval()
    example_lightness = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example_lightness,
        output,
        input_names=["l_channel"],
        output_names=["ab_channels"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

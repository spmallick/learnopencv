#!/usr/bin/env python3
"""Plot channel-pair densities for Rubik's Cube color samples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

from color_spaces import PROJECT_DIR, convert_bgr, read_bgr

# A non-interactive backend keeps validation and CI independent of a desktop.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VALID_PREFIXES = ("blue1", "blue2", "green", "orange1", "orange2", "red", "yellow")


def sample_files(pieces_dir: str | Path, prefix: str) -> list[Path]:
    """Return the deterministically sorted sample files for one cube color."""

    if prefix not in VALID_PREFIXES:
        raise ValueError(f"unknown color prefix {prefix!r}")
    directory = Path(pieces_dir).expanduser().resolve()
    files = sorted(directory.glob(f"{prefix}r*.jpg"))
    if not files:
        raise FileNotFoundError(f"no {prefix}r*.jpg samples found in {directory}")
    return files


def collect_channels(files: Sequence[Path]) -> dict[str, np.ndarray]:
    """Convert all samples once and concatenate their channel values."""

    images = [read_bgr(path) for path in files]
    converted = {
        "BGR": [image for image in images],
        "HSV": [convert_bgr(image, "HSV") for image in images],
        "YCrCb": [convert_bgr(image, "YCrCb") for image in images],
        "Lab": [convert_bgr(image, "Lab") for image in images],
    }

    channels: dict[str, np.ndarray] = {}
    for name, space_images in converted.items():
        pixels = np.concatenate([image.reshape(-1, 3) for image in space_images], axis=0)
        for index in range(3):
            channels[f"{name}{index}"] = pixels[:, index]
    return channels


def create_density_figure(
    channels: dict[str, np.ndarray],
    *,
    bins: int = 20,
    zoom: bool = False,
) -> plt.Figure:
    """Create the six comparison plots used by the tutorial."""

    if bins < 2:
        raise ValueError("bins must be at least 2")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plots = (
        ("BGR0", "BGR1", "B vs G", (0, 255), (0, 255)),
        ("BGR0", "BGR2", "B vs R", (0, 255), (0, 255)),
        ("BGR2", "BGR1", "R vs G", (0, 255), (0, 255)),
        ("HSV0", "HSV1", "Hue vs Saturation", (0, 179), (0, 255)),
        ("YCrCb1", "YCrCb2", "Cr vs Cb", (0, 255), (0, 255)),
        ("Lab1", "Lab2", "Lab a vs b", (0, 255), (0, 255)),
    )

    for axis, (x_name, y_name, title, x_range, y_range) in zip(axes.flat, plots):
        histogram = axis.hist2d(
            channels[x_name],
            channels[y_name],
            bins=bins,
            norm=LogNorm(),
        )
        axis.set_xlabel(x_name)
        axis.set_ylabel(y_name)
        axis.set_title(title)
        if not zoom:
            axis.set_xlim(x_range)
            axis.set_ylim(y_range)
        figure.colorbar(histogram[3], ax=axis)
    return figure


def save_density_plot(
    files: Sequence[Path],
    output: str | Path,
    *,
    bins: int = 20,
    zoom: bool = False,
) -> tuple[Path, int]:
    """Generate, save, verify, and close one density plot."""

    channels = collect_channels(files)
    pixel_count = int(channels["BGR0"].size)
    figure = create_density_figure(channels, bins=bins, zoom=zoom)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)

    verified = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if verified is None or verified.size == 0:
        raise OSError(f"could not verify density plot: {output_path}")
    return output_path, pixel_count


def run_validation(output: str | Path | None = None) -> dict[str, int]:
    """Validate the eight yellow samples and optional rendered plot."""

    files = sample_files(PROJECT_DIR / "pieces", "yellow")
    if len(files) != 8:
        raise AssertionError(f"expected 8 yellow samples, found {len(files)}")
    channels = collect_channels(files)
    pixel_count = int(channels["BGR0"].size)
    if pixel_count <= 0 or any(channel.size != pixel_count for channel in channels.values()):
        raise AssertionError("channel arrays have inconsistent sizes")

    if output is not None:
        output_path, rendered_pixels = save_density_plot(files, output, bins=20, zoom=True)
        if rendered_pixels != pixel_count:
            raise AssertionError("rendered pixel count differs from channel data")
        print(f"Wrote: {output_path}")

    print(f"VALIDATION PASSED: samples={len(files)}, pixels={pixel_count}")
    return {"samples": len(files), "pixels": pixel_count}


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pieces-dir", type=Path, default=PROJECT_DIR / "pieces")
    parser.add_argument("--color", choices=VALID_PREFIXES, default="yellow")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--output", type=Path, help="output PNG path")
    parser.add_argument("--validate", action="store_true", help="run deterministic checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the requested density plot."""

    arguments = _build_parser().parse_args(argv)
    if arguments.validate:
        run_validation(arguments.output)
        return 0

    files = sample_files(arguments.pieces_dir, arguments.color)
    output = arguments.output or Path(f"{arguments.color}{'-zoom' if arguments.zoom else ''}.png")
    output_path, pixel_count = save_density_plot(
        files,
        output,
        bins=arguments.bins,
        zoom=arguments.zoom,
    )
    print(f"Samples: {len(files)}; pixels: {pixel_count}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

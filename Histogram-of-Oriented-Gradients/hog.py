#!/usr/bin/env python3
"""Compute and visualize a 64x128 Histogram of Oriented Gradients descriptor."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class HOGConfig:
    window: tuple[int, int] = (64, 128)
    block: tuple[int, int] = (16, 16)
    block_stride: tuple[int, int] = (8, 8)
    cell: tuple[int, int] = (8, 8)
    bins: int = 9

    @property
    def descriptor_length(self) -> int:
        blocks_x = (self.window[0] - self.block[0]) // self.block_stride[0] + 1
        blocks_y = (self.window[1] - self.block[1]) // self.block_stride[1] + 1
        cells_per_block = (
            self.block[0] // self.cell[0]
        ) * (self.block[1] // self.cell[1])
        return blocks_x * blocks_y * cells_per_block * self.bins


DEFAULT_CONFIG = HOGConfig()


def make_demo_image(config: HOGConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Create a deterministic high-contrast silhouette for a self-contained demo."""
    width, height = config.window
    image = np.full((height, width, 3), 28, dtype=np.uint8)
    cv2.circle(image, (width // 2, 20), 9, (225, 225, 225), -1)
    cv2.ellipse(
        image, (width // 2, 54), (13, 25), 0, 0, 360, (205, 205, 205), -1
    )
    cv2.line(image, (24, 44), (8, 76), (220, 220, 220), 7, cv2.LINE_AA)
    cv2.line(image, (40, 44), (56, 72), (220, 220, 220), 7, cv2.LINE_AA)
    cv2.line(image, (27, 74), (20, 119), (230, 230, 230), 9, cv2.LINE_AA)
    cv2.line(image, (37, 74), (47, 119), (230, 230, 230), 9, cv2.LINE_AA)
    cv2.rectangle(image, (0, 120), (width - 1, height - 1), (75, 75, 75), -1)
    return image


def prepare_image(
    image: np.ndarray, config: HOGConfig = DEFAULT_CONFIG
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if image.ndim not in (2, 3):
        raise ValueError("Input image must be grayscale, BGR, or BGRA")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("Color input must have three or four channels")
    return cv2.resize(image, config.window, interpolation=cv2.INTER_AREA)


def compute_descriptor(
    image: np.ndarray, config: HOGConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Compute a version-stable educational HOG descriptor with L2-Hys blocks."""
    prepared = prepare_image(image, config)
    histograms = cell_histograms(prepared, config)
    block_cells_x = config.block[0] // config.cell[0]
    block_cells_y = config.block[1] // config.cell[1]
    stride_cells_x = config.block_stride[0] // config.cell[0]
    stride_cells_y = config.block_stride[1] // config.cell[1]
    blocks: list[np.ndarray] = []
    epsilon = 1e-5

    for cell_y in range(
        0, histograms.shape[0] - block_cells_y + 1, stride_cells_y
    ):
        for cell_x in range(
            0, histograms.shape[1] - block_cells_x + 1, stride_cells_x
        ):
            block = histograms[
                cell_y : cell_y + block_cells_y,
                cell_x : cell_x + block_cells_x,
            ].reshape(-1)
            block = block / np.sqrt(np.dot(block, block) + epsilon * epsilon)
            block = np.minimum(block, 0.2)
            block = block / np.sqrt(np.dot(block, block) + epsilon * epsilon)
            blocks.append(block)

    descriptor = np.concatenate(blocks).astype(np.float32)
    if descriptor.size != config.descriptor_length:
        raise RuntimeError(
            f"Expected {config.descriptor_length} values, got {descriptor.size}"
        )
    return descriptor


def cell_histograms(
    image: np.ndarray, config: HOGConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Build unnormalized cell histograms for an interpretable visualization."""
    prepared = prepare_image(image, config)
    if prepared.ndim == 3:
        gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
    else:
        gray = prepared
    gray_float = gray.astype(np.float32)
    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=1)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(
        gradient_x, gradient_y, angleInDegrees=True
    )
    angle %= 180.0

    cell_width, cell_height = config.cell
    cells_x = config.window[0] // cell_width
    cells_y = config.window[1] // cell_height
    histograms = np.zeros((cells_y, cells_x, config.bins), dtype=np.float32)
    bin_width = 180.0 / config.bins

    for cell_y in range(cells_y):
        y0 = cell_y * cell_height
        for cell_x in range(cells_x):
            x0 = cell_x * cell_width
            cell_angles = angle[y0 : y0 + cell_height, x0 : x0 + cell_width]
            cell_magnitudes = magnitude[
                y0 : y0 + cell_height, x0 : x0 + cell_width
            ]
            bin_position = cell_angles / bin_width
            lower = np.floor(bin_position).astype(np.int32) % config.bins
            upper = (lower + 1) % config.bins
            upper_weight = bin_position - np.floor(bin_position)
            lower_weight = 1.0 - upper_weight
            for bin_index in range(config.bins):
                histograms[cell_y, cell_x, bin_index] = np.sum(
                    cell_magnitudes
                    * (
                        (lower == bin_index) * lower_weight
                        + (upper == bin_index) * upper_weight
                    )
                )
    return histograms


def visualize_hog(
    image: np.ndarray,
    config: HOGConfig = DEFAULT_CONFIG,
    *,
    scale: int = 4,
) -> np.ndarray:
    prepared = prepare_image(image, config)
    if prepared.ndim == 2:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
    elif prepared.shape[2] == 4:
        prepared = cv2.cvtColor(prepared, cv2.COLOR_BGRA2BGR)

    canvas = cv2.resize(
        prepared, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    canvas = cv2.addWeighted(
        canvas, 0.45, np.zeros_like(canvas), 0.55, 0
    )
    histograms = cell_histograms(prepared, config)
    maximum = float(histograms.max())
    if maximum <= 0:
        return canvas

    cell_width, cell_height = config.cell
    line_radius = min(cell_width, cell_height) * scale * 0.42
    for cell_y in range(histograms.shape[0]):
        for cell_x in range(histograms.shape[1]):
            center = (
                int((cell_x + 0.5) * cell_width * scale),
                int((cell_y + 0.5) * cell_height * scale),
            )
            for bin_index, value in enumerate(histograms[cell_y, cell_x]):
                strength = float(value) / maximum
                if strength < 0.03:
                    continue
                # Voting treats bin 0 as centered at 0 degrees, bin 1 at
                # 20 degrees, and so on. Draw those same centers.
                angle = np.deg2rad(bin_index * 180.0 / config.bins)
                dx = np.cos(angle) * line_radius * strength
                dy = np.sin(angle) * line_radius * strength
                start = (int(round(center[0] - dx)), int(round(center[1] - dy)))
                end = (int(round(center[0] + dx)), int(round(center[1] + dy)))
                cv2.line(canvas, start, end, (50, 230, 255), 1, cv2.LINE_AA)
    return canvas


def run(
    input_path: Path | None,
    output_dir: Path,
    config: HOGConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    if input_path is None:
        image = make_demo_image(config)
        input_source = "generated-demo"
    else:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read input image: {input_path}")
        input_source = str(input_path)

    prepared = prepare_image(image, config)
    descriptor = compute_descriptor(prepared, config)
    visualization = visualize_hog(prepared, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_output = output_dir / "hog-input.png"
    visualization_output = output_dir / "hog-visualization.png"
    descriptor_output = output_dir / "hog-descriptor.npy"
    if not cv2.imwrite(str(input_output), prepared):
        raise RuntimeError(f"Could not write: {input_output}")
    if not cv2.imwrite(str(visualization_output), visualization):
        raise RuntimeError(f"Could not write: {visualization_output}")
    np.save(descriptor_output, descriptor)

    blocks_x = (
        config.window[0] - config.block[0]
    ) // config.block_stride[0] + 1
    blocks_y = (
        config.window[1] - config.block[1]
    ) // config.block_stride[1] + 1
    cells_per_block = (
        config.block[0] // config.cell[0]
    ) * (config.block[1] // config.cell[1])

    return {
        "input": input_source,
        "window": list(config.window),
        "blocks": [blocks_x, blocks_y],
        "cells_per_block": cells_per_block,
        "bins": config.bins,
        "descriptor_length": int(descriptor.size),
        "descriptor_l2_norm": float(np.linalg.norm(descriptor)),
        "outputs": {
            "input": str(input_output),
            "visualization": str(visualization_output),
            "descriptor": str(descriptor_output),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args.input, args.output_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

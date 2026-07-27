#!/usr/bin/env python3
"""Inspect a BGR pixel in HSV, YCrCb, and Lab encodings."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from color_spaces import (
    convert_pixel,
    read_bgr,
    resolve_input_path,
    run_core_validation,
    write_image,
)

WINDOW_NAME = "Move the mouse over the image; press Esc to exit"


def render_pixel_panel(image: np.ndarray, x: int, y: int) -> np.ndarray:
    """Render the image beside a text panel describing one valid pixel."""

    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"pixel ({x}, {y}) is outside image bounds {width}x{height}")

    values = convert_pixel(image[y, x])
    panel = np.zeros((height, 420, 3), dtype=np.uint8)
    lines = [
        f"Pixel (x={x}, y={y})",
        f"BGR   {values['BGR']}",
        f"HSV   {values['HSV']}",
        f"YCrCb {values['YCrCb']}",
        f"Lab   {values['Lab']}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (20, 45 + index * 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    marked = image.copy()
    cv2.drawMarker(marked, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    return np.hstack((marked, panel))


def run_validation(output: str | Path | None = None) -> dict[str, int]:
    """Run deterministic dataset and rendering checks."""

    metrics = run_core_validation()
    image = read_bgr(resolve_input_path(None))
    x_coordinate = image.shape[1] // 2
    y_coordinate = image.shape[0] // 2
    rendered = render_pixel_panel(image, x_coordinate, y_coordinate)
    if rendered.shape != (image.shape[0], image.shape[1] + 420, 3):
        raise AssertionError(f"unexpected rendered shape: {rendered.shape}")
    if output is not None:
        write_image(output, rendered)

    print(
        "VALIDATION PASSED: "
        f"{metrics['cube_images']} cube images, "
        f"{metrics['piece_images']} piece images, "
        f"panel={rendered.shape[1]}x{rendered.shape[0]}"
    )
    return {
        **metrics,
        "output_width": rendered.shape[1],
        "output_height": rendered.shape[0],
    }


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="input image; defaults to images/rub00.jpg")
    parser.add_argument("--x", type=int, help="sample x coordinate; defaults to image center")
    parser.add_argument("--y", type=int, help="sample y coordinate; defaults to image center")
    parser.add_argument("--output", type=Path, help="optional annotated output image")
    parser.add_argument("--no-display", action="store_true", help="do not open a GUI window")
    parser.add_argument("--validate", action="store_true", help="run deterministic checks")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI or interactive pixel inspector."""

    arguments = _build_parser().parse_args(argv)
    if arguments.validate:
        run_validation(arguments.output)
        return 0

    image_path = resolve_input_path(arguments.input)
    image = read_bgr(image_path)
    x_coordinate = arguments.x if arguments.x is not None else image.shape[1] // 2
    y_coordinate = arguments.y if arguments.y is not None else image.shape[0] // 2
    rendered = render_pixel_panel(image, x_coordinate, y_coordinate)

    values = convert_pixel(image[y_coordinate, x_coordinate])
    print(f"Input: {image_path}")
    print(f"Pixel: x={x_coordinate}, y={y_coordinate}")
    for name, value in values.items():
        print(f"{name}: {value}")

    if arguments.output is not None:
        output_path = write_image(arguments.output, rendered)
        print(f"Wrote: {output_path}")

    if arguments.no_display:
        return 0

    state = {"image": image, "rendered": rendered}

    def on_mouse(event: int, x_value: int, y_value: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_MOUSEMOVE:
            return
        if 0 <= x_value < image.shape[1] and 0 <= y_value < image.shape[0]:
            state["rendered"] = render_pixel_panel(image, x_value, y_value)
            cv2.imshow(WINDOW_NAME, state["rendered"])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    cv2.imshow(WINDOW_NAME, state["rendered"])
    while cv2.waitKey(20) & 0xFF != 27:
        pass
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

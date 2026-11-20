"""Compose a LearnOpenCV thumbnail with OpenCV's built-in drawing functions.

Photographs come from the recorded Graf6 result. Typography, frames, and an
explanatory arrow are added without inventing matches or alignment results.
Run python scripts/make_thumbnail.py; external font files are not needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SIZE = (1600, 900)  # Width and height match the site's featured-image convention.
# OpenCV drawing and image I/O use BGR channel order throughout this script.
NAVY = (28, 15, 7)
PANEL = (47, 29, 17)
WHITE = (252, 250, 248)
MUTED = (232, 218, 206)
ACCENT = (226, 193, 32)


def rounded_rectangle(canvas: np.ndarray, bounds: tuple[int, int, int, int],
                      color: tuple[int, int, int], radius: int) -> None:
    """Join filled rectangles and antialiased circles into a rounded panel."""
    left, top, right, bottom = bounds
    cv2.rectangle(canvas, (left + radius, top), (right - radius, bottom), color, -1)
    cv2.rectangle(canvas, (left, top + radius), (right, bottom - radius), color, -1)
    for x in (left + radius, right - radius):
        for y in (top + radius, bottom - radius):
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)


def text(canvas: np.ndarray, xy: tuple[int, int], value: str, height: int,
         color: tuple[int, int, int], *, max_width: int | None = None,
         title: bool = False) -> int:
    """Size built-in text by visible height and anchor it at the top left."""
    font = cv2.FONT_HERSHEY_DUPLEX if title else cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3 if title else 2
    (_, unit_height), _ = cv2.getTextSize(value, font, 1.0, thickness)
    scale = height / unit_height
    (width, _), _ = cv2.getTextSize(value, font, scale, thickness)
    if max_width is not None and width > max_width:
        scale *= max_width / width
    (width, visible_height), _ = cv2.getTextSize(value, font, scale, thickness)
    # putText takes a baseline origin; convert the requested top-left anchor.
    cv2.putText(canvas, value, (xy[0], xy[1] + visible_height), font, scale,
                color, thickness, cv2.LINE_AA)
    return width


def render() -> np.ndarray:
    source_path = PROJECT / 'figures/graf_alignment.png'
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise OSError(f'Cannot read thumbnail source: {source_path}')
    # These reviewed crops retain the actual photographs and aligned overlay.
    if source.shape[:2] != (728, 2085):
        raise ValueError('Graf result layout changed; review photograph-panel crop boxes.')
    moving = source[162:576, 565:1082]
    aligned = source[162:576, 1106:1623]
    canvas = np.full((SIZE[1], SIZE[0], 3), NAVY, dtype=np.uint8)

    badge = 'LearnOpenCV'
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, unit_height), _ = cv2.getTextSize(badge, font, 1.0, 2)
    (badge_width, _), _ = cv2.getTextSize(badge, font, 30 / unit_height, 2)
    rounded_rectangle(canvas, (72, 42, 72 + badge_width + 36, 98), ACCENT, 12)
    text(canvas, (90, 53), badge, 30, NAVY)
    text(canvas, (72, 133), 'Robust Image Alignment', 88, WHITE,
         max_width=1456, title=True)
    # Hershey fonts support ASCII, so use compatible separators.
    text(canvas, (75, 243), 'ALIKED  |  LightGlue  |  MAGSAC++', 40, MUTED,
         max_width=1450)
    rounded_rectangle(canvas, (74, 297, 174, 304), ACCENT, 3)

    for x, label, photo in [(72, 'VIEWPOINT CHANGE', moving), (878, 'ALIGNED OVERLAY', aligned)]:
        rounded_rectangle(canvas, (x, 341, x + 650, 850), (96, 73, 52), 14)
        rounded_rectangle(canvas, (x + 2, 343, x + 648, 848), PANEL, 12)
        text(canvas, (x + 22, 359), label, 28, WHITE, max_width=606)
        # Both crops share one scale and retain their original aspect ratio.
        scale = min(626 / photo.shape[1], 432 / photo.shape[0])
        width, height = round(photo.shape[1] * scale), round(photo.shape[0] * scale)
        fitted = cv2.resize(photo, (width, height), interpolation=cv2.INTER_LANCZOS4)
        left, top = x + (650 - width) // 2, 405 + (432 - height) // 2
        canvas[top:top + height, left:left + width] = fitted

    cv2.circle(canvas, (800, 603), 50, ACCENT, -1, cv2.LINE_AA)
    cv2.arrowedLine(canvas, (774, 603), (829, 603), NAVY, 8, cv2.LINE_AA, tipLength=0.4)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'figures',
                        help='Directory for the PNG and WebP thumbnail exports.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canvas = render()
    output = args.output_dir / 'robust-image-alignment-thumbnail'
    for suffix, options in [('.png', [cv2.IMWRITE_PNG_COMPRESSION, 9]),
                            ('.webp', [cv2.IMWRITE_WEBP_QUALITY, 93])]:
        destination = output.with_suffix(suffix)
        if not cv2.imwrite(str(destination), canvas, options):
            raise OSError(f'Cannot write thumbnail: {destination}')
    print(f'Saved {output.name}.png and .webp ({SIZE[0]}x{SIZE[1]}).')


if __name__ == '__main__':
    main()

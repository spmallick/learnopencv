"""Compose a LearnOpenCV thumbnail from the article's actual recorded result.

Only typography, frames, and an explanatory arrow are added. The photograph
and aligned overlay are copied from the tracked Graf6 figure; no image content,
matches, measurements, or successful outcome is invented for the thumbnail.
Run with Pillow: ``python scripts/make_thumbnail.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
SIZE = (1600, 900)  # Matches the current LearnOpenCV 16:9 featured-image convention.
NAVY = (7, 15, 28)
PANEL = (17, 29, 47)
WHITE = (248, 250, 252)
MUTED = (206, 218, 232)
ACCENT = (32, 193, 226)


def font_file(name: str, override: str | None) -> Path:
    """Prefer the site's existing Arial typography; accept a portable override."""
    if override:
        result = Path(override)
    else:
        result = Path('/System/Library/Fonts/Supplemental') / name
    if not result.is_file():
        raise FileNotFoundError(f'Font not found: {result}; supply --title-font and --body-font.')
    return result


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str,
         font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
    """Use top-left anchoring so visual bounds agree with the margin calculation."""
    draw.text(xy, value, font=font, fill=fill, anchor='lt')


def render(title_path: Path, body_path: Path) -> Image.Image:
    source_path = PROJECT / 'figures/graf_alignment.png'
    with Image.open(source_path) as image:
        source = image.convert('RGB')
    # These bounds remove plotting labels/margins while retaining the full panels.
    # A changed source layout requires an explicit visual review of the crops.
    if source.size != (2085, 728):
        raise ValueError('Graf result layout changed; review the photograph-panel crop boxes.')
    moving = source.crop((565, 162, 1082, 576))
    aligned = source.crop((1106, 162, 1623, 576))
    canvas = Image.new('RGB', SIZE, NAVY)
    draw = ImageDraw.Draw(canvas)

    badge_font = ImageFont.truetype(str(body_path), 30)
    title_font = ImageFont.truetype(str(title_path), 88)
    subtitle_font = ImageFont.truetype(str(body_path), 40)
    label_font = ImageFont.truetype(str(body_path), 28)

    # Branding is a text badge matching the existing LearnOpenCV thumbnail style.
    badge = 'LearnOpenCV'
    badge_width = round(draw.textlength(badge, font=badge_font))
    draw.rounded_rectangle((72, 42, 72 + badge_width + 36, 98), radius=12, fill=ACCENT)
    text(draw, (90, 53), badge, badge_font, NAVY)

    headline = 'Robust Image Alignment'
    if draw.textlength(headline, font=title_font) > 1456:
        raise ValueError('The requested title font exceeds the safe headline width.')
    text(draw, (72, 133), headline, title_font, WHITE)
    text(draw, (75, 243), 'ALIKED  •  LightGlue  •  MAGSAC++', subtitle_font, MUTED)
    draw.rounded_rectangle((74, 297, 174, 304), radius=3, fill=ACCENT)

    # The two photographs share the same scale; aspect ratio is preserved.
    for x, label, photo in [(72, 'VIEWPOINT CHANGE', moving), (878, 'ALIGNED OVERLAY', aligned)]:
        draw.rounded_rectangle((x, 341, x + 650, 850), radius=14, fill=PANEL,
                               outline=(52, 73, 96), width=2)
        text(draw, (x + 22, 359), label, label_font, WHITE)
        fitted = ImageOps.contain(photo, (626, 432), Image.Resampling.LANCZOS)
        left = x + (650 - fitted.width) // 2
        top = 405 + (432 - fitted.height) // 2
        canvas.paste(fitted, (left, top))

    # This arrow expresses the operation; it is not a fabricated match overlay.
    draw.ellipse((750, 553, 850, 653), fill=ACCENT)
    draw.line((774, 603, 826, 603), fill=NAVY, width=10)
    draw.line((807, 583, 829, 603, 807, 623), fill=NAVY, width=10,
              joint='curve')
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--title-font', help='Path to Arial Black or a comparable bold font.')
    parser.add_argument('--body-font', help='Path to Arial Bold or a comparable bold font.')
    args = parser.parse_args()
    title_path = font_file('Arial Black.ttf', args.title_font)
    body_path = font_file('Arial Bold.ttf', args.body_font)
    canvas = render(title_path, body_path)
    output = PROJECT / 'figures/robust-image-alignment-thumbnail'
    canvas.save(output.with_suffix('.png'), optimize=True)
    canvas.save(output.with_suffix('.webp'), quality=93, method=6)
    print(f'Saved {output.name}.png and .webp ({canvas.width}×{canvas.height}).')


if __name__ == '__main__':
    main()

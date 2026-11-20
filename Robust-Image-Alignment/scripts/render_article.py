"""Render the editable Markdown draft as a local, readable HTML preview."""

from pathlib import Path
from html import escape
import json
import re
import xml.etree.ElementTree as ET

import markdown
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
source = (PROJECT / 'article.md').read_text()
source = re.sub(r'<!-- repository-reader-note -->.*?<!-- /repository-reader-note -->\s*', '', source, flags=re.S)
# GitHub uses dollar-delimited display math; the local preview uses MathJax.
source = re.sub(r'\$\$(.*?)\$\$', lambda match: r'\[' + match.group(1) + r'\]', source, flags=re.S)
# Preserve LaTeX display delimiters before Markdown treats backslashes as escapes.
source = re.sub(r'\\\[(.*?)\\\]', lambda match: '<div class="math-block">\\[\n' + match.group(1) + '\n\\]</div>', source, flags=re.S)
# Compact WebP figures are equivalent exports of the retained publication PNGs.
for name in re.findall(r'\]\((figures/[^)]+\.png)\)', source):
    alternative = Path(name).with_suffix('.webp')
    if (PROJECT / alternative).exists():
        source = source.replace(f']({name})', f']({alternative.as_posix()})')
body = markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc', 'sane_lists'])


def size_images(fragment):
    """Reserve layout space and prioritize only the first, above-fold image."""
    first = True

    def decorate(match):
        nonlocal first
        tag = match.group(0)
        source_match = re.search(r'src="([^"]+)"', tag)
        if not source_match:
            return tag
        path = PROJECT / source_match.group(1)
        if not path.is_file():
            return tag
        if path.suffix == '.svg':
            root = ET.parse(path).getroot()
            viewbox = root.get('viewBox', '').split()
            if len(viewbox) == 4:
                width, height = (round(float(value)) for value in viewbox[2:])
            else:
                # SVG dimensions may use pt or px; their ratio reserves layout space.
                width = round(float(re.match(r'[\d.]+', root.get('width', '0'))[0]))
                height = round(float(re.match(r'[\d.]+', root.get('height', '0'))[0]))
        else:
            with Image.open(path) as image:
                width, height = image.size
        attrs = f' width="{width}" height="{height}" decoding="async"'
        attrs += ' loading="eager" fetchpriority="high"' if first else ' loading="lazy"'
        first = False
        return re.sub(r'\s*/?>$', '', tag) + attrs + ' />'

    return re.sub(r'<img\b[^>]*>', decorate, fragment)


body = size_images(body)
seo_path = PROJECT / 'research' / 'seo-metadata.json'
seo = json.loads(seo_path.read_text()) if seo_path.exists() else {}
description = escape(seo.get('meta_description', ''), quote=True)
title = escape(source.splitlines()[0].removeprefix('# ').strip(), quote=True)
head_metadata = f'<meta name="description" content="{description}">\n'
head_metadata += f'<meta property="og:title" content="{title}">\n'
head_metadata += f'<meta property="og:description" content="{description}">\n'
head_metadata += '<meta property="og:type" content="article">\n<meta name="twitter:card" content="summary_large_image">\n'
if seo.get('canonical_url'):
    canonical = escape(seo['canonical_url'], quote=True)
    head_metadata += f'<link rel="canonical" href="{canonical}">\n<meta property="og:url" content="{canonical}">\n'
if seo.get('featured_image_public_url'):
    featured = escape(seo['featured_image_public_url'], quote=True)
    head_metadata += f'<meta property="og:image" content="{featured}">\n'
html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robust Image Alignment with ALIKED, LightGlue, and MAGSAC++</title>
''' + head_metadata + '''
<style>
:root{color-scheme:light;--ink:#17283f;--muted:#53647a;--line:#dae2e9;--accent:#087ea4}
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:var(--ink);font:18px/1.72 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1120px;margin:32px auto;background:white;padding:54px 72px;border:1px solid var(--line);border-radius:10px}
h1{font-size:2.55rem;line-height:1.16;letter-spacing:-.035em;margin:0 0 32px}h2{font-size:1.7rem;line-height:1.3;margin-top:58px;border-top:1px solid var(--line);padding-top:26px}h3{font-size:1.25rem;margin-top:34px}
p,ul,ol{max-width:840px}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:28px auto 12px}p:has(>em:only-child){font-size:.9em;color:var(--muted);line-height:1.5;margin:10px auto 30px}
pre{overflow-x:auto;background:#112136;color:#edf5ff;border-radius:7px;padding:20px 24px;line-height:1.55;font-size:14px}code{font-family:"SFMono-Regular",Consolas,monospace}p code,li code,td code{background:#eef3f7;padding:2px 5px;border-radius:4px;font-size:.88em}
table{border-collapse:collapse;display:block;overflow-x:auto;width:100%;font-size:15px;margin:26px 0}th,td{padding:11px 14px;border:1px solid var(--line);text-align:left;vertical-align:top}th{background:#edf3f7}blockquote{margin-left:0;border-left:4px solid var(--accent);padding-left:20px;color:var(--muted)}
@media(max-width:760px){body{font-size:16px}main{padding:28px 20px;margin:0;border:0;border-radius:0}h1{font-size:2rem}h2{font-size:1.4rem;margin-top:38px}pre{font-size:12px;padding:14px}table{font-size:13px}}
@media print{body{background:white}main{border:0;margin:0;padding:0;max-width:none}h2{break-after:avoid}img,pre{break-inside:avoid}a{color:inherit}}
</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head><body><main><article>''' + body + '</article></main></body></html>\n'
(PROJECT / 'article.html').write_text(html)
print('Rendered', PROJECT / 'article.html')

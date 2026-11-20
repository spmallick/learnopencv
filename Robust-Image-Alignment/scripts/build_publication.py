"""Prepare the WordPress article fragment and media from the reviewed source."""

from __future__ import annotations

import argparse
from html import escape, unescape
import json
from pathlib import Path
import re
import shutil

import markdown
import cv2


PROJECT = Path(__file__).resolve().parents[1]


def prepare(output: Path) -> None:
    assets = output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    source = (PROJECT / 'article.md').read_text()
    # GitHub readers need a link to the site; the site already supplies its button.
    source = re.sub(r'<!-- repository-reader-note -->.*?<!-- /repository-reader-note -->\s*', '', source, flags=re.S)
    seo = json.loads((PROJECT / 'research/seo-metadata.json').read_text())

    # Fresh names keep browser and WordPress caches of the original diagrams valid.
    revised_figures = {'pipeline', 'aliked_sampling', 'lightglue_depth',
                       'benchmark_grid_errors', 'magsac_threshold_sensitivity'}

    def copy_image(match):
        label, relative = match.groups()
        original = PROJECT / relative
        # Existing PNG exports preserve the explanatory SVGs in a WP-safe format.
        raster = original.with_suffix('.png') if original.suffix == '.svg' else original
        webp = raster.with_suffix('.webp')
        published_name = f'{webp.stem}-20261120.webp' if webp.stem in revised_figures else webp.name
        destination = assets / published_name
        if webp.exists():
            shutil.copy2(webp, destination)
        else:
            image = cv2.imread(str(raster), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                raise OSError(f'Cannot read publication image: {raster}')
            if image.dtype.name != 'uint8':
                raise ValueError(f'Expected an 8-bit publication image: {raster}')
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                # PNG diagrams may carry alpha; preserve their intended white page.
                alpha = image[..., 3:4].astype('float32') / 255.0
                image = (image[..., :3].astype('float32') * alpha + 255.0 * (1.0 - alpha)).round().astype('uint8')
            elif image.shape[2] != 3:
                raise ValueError(f'Expected a gray, BGR, or BGRA publication image: {raster}')
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_WEBP_QUALITY, 92]):
                raise OSError(f'Cannot write publication image: {destination}')
        return f'![{label}](assets/{destination.name})'

    source = re.sub(r'!\[([^\]]*)\]\((figures/[^)]+)\)', copy_image, source)
    featured = PROJECT / 'figures' / Path(seo['featured_image_absolute_path']).name
    featured_copy = assets / featured.name
    shutil.copy2(featured, featured_copy)
    slug = seo['proposed_slug']
    markdown_path = output / f'{slug}.md'
    markdown_path.write_text(source)

    # Plain HTML mathematics renders without depending on a site-wide MathJax script.
    equation = ('<div class="alignment-equation" role="math" '
                'aria-label="Homogeneous image coordinates and perspective division">'
                '<p><strong>[u v w]<sup>T</sup> = H [x y 1]<sup>T</sup>, '
                '&nbsp; x′ = u/w, &nbsp; y′ = v/w.</strong></p></div>')
    rendered_source = re.sub(r'\\\[.*?\\\]|\$\$.*?\$\$', lambda _: equation, source, flags=re.S)
    fragment = markdown.markdown(rendered_source, extensions=['tables', 'fenced_code', 'toc', 'sane_lists'])

    def code_block(match):
        language, code = match.groups()
        language = language or 'plain'
        language = 'plain' if language in {'bash', 'sh', 'shell', 'text'} else language
        return (f'<!-- wp:syntaxhighlighter/code {{"language":"{language}"}} -->\n'
                f'<pre class="wp-block-syntaxhighlighter-code">{code}</pre>\n'
                '<!-- /wp:syntaxhighlighter/code -->')

    fragment = re.sub(r'<pre><code(?: class="language-([^\"]+)")?>(.*?)</code></pre>',
                      code_block, fragment, flags=re.S)
    first_image = True

    def image_tag(match):
        nonlocal first_image
        tag = match.group(0)
        relative = unescape(re.search(r'src="([^\"]+)"', tag)[1])
        image_path = output / relative
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            raise OSError(f'Cannot read publication image: {image_path}')
        height, width = image.shape[:2]
        loading = 'loading="eager" fetchpriority="high"' if first_image else 'loading="lazy"'
        first_image = False
        return (re.sub(r'\s*/?>$', '', tag) + f' width="{width}" height="{height}" decoding="async" '
                + loading + ' style="max-width:100%;height:auto;" />')

    fragment = re.sub(r'<img\b[^>]*>', image_tag, fragment)
    fragment = re.sub(r'<table>(.*?)</table>',
                      r'<div role="region" aria-label="Scrollable comparison table" tabindex="0" '
                      r'style="max-width:100%;overflow-x:auto;"><table>\1</table></div>',
                      fragment, flags=re.S)
    (output / f'{slug}.html').write_text(f'<article>\n{fragment}\n</article>\n')
    # The theme supplies the post's H1, while the complete handoff retains its title.
    wordpress_fragment = re.sub(r'<h1\b[^>]*>.*?</h1>\s*', '', fragment, count=1, flags=re.S)
    (output / f'{slug}.wp.html').write_text(f'<article>\n{wordpress_fragment}\n</article>\n')
    values = {
        'SEO title': seo['seo_title'], 'Slug': slug,
        'Meta description': seo['meta_description'],
        'Featured image': str(featured_copy.resolve()),
        'Primary keyphrase': seo['primary_keyphrase'],
        'Secondary keyphrases': ', '.join(seo['secondary_keyphrases']),
        'WordPress categories': 'Image Alignment, OpenCV 5, OpenCV Tutorials',
    }
    (output / f'{slug}-seo.md').write_text('\n'.join(f'{key}: {value}' for key, value in values.items()) + '\n')
    metadata_path = PROJECT / 'publication/post-metadata.json'
    if metadata_path.exists():
        shutil.copy2(metadata_path, output / 'post-metadata.json')
    print(f'Prepared WordPress Markdown, HTML, SEO, and {len(list(assets.iterdir()))} media files in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir.resolve())

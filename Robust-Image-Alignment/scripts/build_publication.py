"""Prepare the WordPress article fragment and media from the reviewed source."""

from __future__ import annotations

import argparse
from html import escape, unescape
import json
from pathlib import Path
import re
import shutil

import markdown
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]


def prepare(output: Path) -> None:
    assets = output / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    source = (PROJECT / 'article.md').read_text()
    # GitHub readers need a link to the site; the site already supplies its button.
    source = re.sub(r'<!-- repository-reader-note -->.*?<!-- /repository-reader-note -->\s*', '', source, flags=re.S)
    seo = json.loads((PROJECT / 'research/seo-metadata.json').read_text())

    def copy_image(match):
        label, relative = match.groups()
        original = PROJECT / relative
        # Existing PNG exports preserve the explanatory SVGs in a WP-safe format.
        raster = original.with_suffix('.png') if original.suffix == '.svg' else original
        webp = raster.with_suffix('.webp')
        destination = assets / webp.name
        if webp.exists():
            shutil.copy2(webp, destination)
        else:
            with Image.open(raster) as image:
                image.convert('RGB').save(destination, 'WEBP', quality=92, method=6)
        return f'![{label}](assets/{destination.name})'

    source = re.sub(r'!\[([^\]]*)\]\((figures/[^)]+)\)', copy_image, source)
    featured = Path(seo['featured_image_absolute_path'])
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
        with Image.open(output / relative) as image:
            width, height = image.size
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
        'WordPress categories': 'Image Alignment, OpenCV Tutorials',
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

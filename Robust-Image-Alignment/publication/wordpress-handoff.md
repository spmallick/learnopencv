# WordPress publication and newsletter download handoff

Article title: **Robust Image Alignment with ALIKED, LightGlue, and MAGSAC++**.
The user authorized GitHub publication and subsequently requested full
LearnOpenCV publication, including its thumbnail. This file documents the
publishing contract; it does not itself change WordPress.

## Article and gated download

Keep the Python teaching excerpts in the article. The complete download includes
the Jupyter notebook, standalone Python implementation, native C++ implementation,
models, tests, setup instructions, and retained evidence. The release must contain
one top-level `Robust-Image-Alignment/` directory and be derived from the tested
Git commit. The exact public GitHub Release ZIP and its SHA-256 must be verified
before configuring the newsletter flow.

The article callout should tell readers to use the site's **Download Code**
control and register. Do not add a repository URL, notebook link or filename,
direct ZIP URL, Google Colab link, download shortcode, or embedded signup form
to article Markdown or WordPress content. The companion repository README may
link its download button to the verified release ZIP.

GitHub cannot run the LearnOpenCV popup. After the LearnOpenCV article is live,
the GitHub article may direct readers to that live article's **Download Code**
control. A generic newsletter link must not be presented as a working delivery
path for this particular ZIP unless its package-specific behavior is verified.

## Configure WordPress metadata

Set the verified GitHub Release asset URL in the ACF **Blog Post → Code URL**
field, separately from the article body:

- Field name: `code_url`
- Field key: `field_605e315832c84`
- WordPress Admin input: `acf[field_605e315832c84]`
- Frontend newsletter hidden input: `form_fields[code_url]`

Use WordPress Admin unless a current capability check proves this ACF field is
writable through an authenticated API. Standard post REST access does not prove
ACF write support. Do not activate a guessed URL, mutable branch archive, or
repository-directory URL.

## Verified site behavior

A read-only inspection on November 19, 2026 of the existing
[feature-based alignment article](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
found Elementor popup `75958`, form name `download_code`, and opening selectors
`.elementorDownloadPopup`, `.github`, and `.top-bar-download`. The popup's hidden
`form_fields[code_url]` contains that article's versioned GitHub Release ZIP.
The navigation's generic **Subscribe** link points to the homepage with `#`;
it is not a package-specific download endpoint. These are observations of an
existing post, not proof that the new article is configured.

## Publication checks

1. Resolve the new article's exact WordPress identity and avoid an existing post
   with the same title or slug. Keep unrelated existing alignment articles intact.
2. Use the SEO-reviewed title, description, slug, categories, and featured image.
   Upload local article media and replace local paths with their verified media
   URLs. The handoff HTML is one `<article>` fragment with SyntaxHighlighter code
   blocks, stable heading IDs, and a matching table of contents.
3. Verify the published GitHub ZIP anonymously, check its SHA-256 and manifest,
   then set **Code URL** to that exact asset. Record it only in metadata.
4. After publication, confirm post ID, slug, title, status, author, featured media,
   content, canonical URL, SEO metadata, dates, images, and internal anchors.
5. Open the site's **Download Code** control and inspect the newsletter popup.
   Confirm its hidden `form_fields[code_url]` equals the verified ZIP URL and that
   name/email registration is still required. **Do not submit the form** during
   verification. This inspection does not test delivery of an actual email.
6. Search the published article body for prohibited code/download URLs and
   notebook references. Verify that the page presents the current article rather
   than cached prior content; do not purge sitewide caches automatically.

The exact live identity, release URL, checksum, and final validation outcomes
belong in [post-metadata.json](post-metadata.json). Unset fields are pending
publication and must not be interpreted as an active website configuration.

An authenticated read-only preflight on November 19 found no post or draft
matching the exact phrase “Robust Image Alignment” or either proposed slug
`robust-image-alignment` / `robust-image-alignment-aliked-lightglue-magsac`.
Existing posts about feature-based alignment and neural feature matching are
separate articles. Repeat this identity check immediately before creating the
new post.

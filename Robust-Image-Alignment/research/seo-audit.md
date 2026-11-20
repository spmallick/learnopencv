# SEO audit: robust image alignment

Audited November 19, 2026, before GitHub and LearnOpenCV publication. The baseline
was local commit `8e432bed`, before the publishing edits. This is an editorial
and technical audit; it does not claim search-volume estimates, a ranking score,
or a measured Core Web Vitals result. Publication and newsletter delivery must
be verified separately after the final files and site are updated.

## Search intent and useful content

The article answers a practical implementation question: how to align images
and decide whether to trust the result. It gives a geometric model, a classical
baseline, learned correspondence extraction/matching, robust estimation,
single-reference localization, rejection checks, independent accuracy, runtime
boundaries, and failure examples. Those tested results and limitations are
substantial original value, consistent with Google's [people-first content
guidance](https://developers.google.com/search/docs/fundamentals/creating-helpful-content).

Keep **Robust Image Alignment with ALIKED, LightGlue, and MAGSAC++** exactly as
requested. It is 59 characters and leads with the general task before naming
the methods. The HTML title and sole H1 already agree. Google recommends a
descriptive, concise [title](https://developers.google.com/search/docs/appearance/title-link);
59 characters is an observed length, not a guarantee about display truncation.

Use `robust image alignment` as the primary editorial keyphrase. Related terms
are `image alignment OpenCV`, `image registration`, `ALIKED LightGlue`,
`MAGSAC++ homography`, and `single-reference object identification`. Baseline
prose naturally contains image alignment 5 times, image registration once,
OpenCV 40 times, ALIKED 22 times, LightGlue 25 times, MAGSAC++ 20 times, and
homography 22 times. These counts are descriptive; no density target or keyword
stuffing is recommended. Do not call the method semantic one-shot learning or
promise a universal best matcher.

## Metadata prepared

- SEO title and social title: the exact requested title.
- Proposed slug: `robust-image-alignment`, subject to checking the site's existing
  posts and final canonical URL. Do not activate a guessed URL.
- Meta description, 148 characters: “Learn robust image alignment with ALIKED,
  LightGlue, and MAGSAC++ in OpenCV. Align documents, locate objects from one
  reference, and check failures.”
- Social description: the same accurate summary.
- Featured image: use the measured perspective-alignment photography, with a
  readable LearnOpenCV thumbnail; retain the images' original asset provenance.

The baseline local HTML lacks a meta description, Open Graph/Twitter metadata,
a canonical link, and Article JSON-LD. Add the description and social metadata
to the standalone preview; use the site's SEO integration for the published
page. A description is a useful candidate for Google's snippet, which Google
may instead generate from the [page content](https://developers.google.com/search/docs/appearance/snippet).

## Changes to make before publication

| Priority | Finding | Required action |
| --- | --- | --- |
| High | The GitHub article's Download Code text is plain bold text. GitHub does not run the LearnOpenCV theme's popup. | Provide a real, verified link/button that leads to the LearnOpenCV newsletter registration flow. The article must never point to the ZIP, release asset, source directory, raw file, notebook, or Python/C++ source. On LearnOpenCV, use the site-managed Download Code button and the verified ACF Code URL. |
| High | No meta description/social metadata is applied to the preview. | Apply the prepared metadata; on the live page, verify the final title, description, social image, and URL in rendered HTML. |
| Medium | The baseline has about 5,258 prose words, thirteen numbered main sections, and no linked contents or learning objectives. | Add concise What You Will Learn and a manually linked contents list. Preserve stable section IDs and verify every fragment target. |
| Medium | The baseline HTML has twelve image elements and no explicit intrinsic dimensions. | Add width and height/aspect ratio to reserve space. Keep the first image eager; later images may be lazy-loaded. Inspect the actual page for layout and legibility. |
| Medium | The raw-experiment link `results/experiments.json` works in the repository but would resolve incorrectly from WordPress. | For WordPress, use an uploaded, verified evidence asset or a plain-text reference to the evidence included in the download. Do not substitute a direct companion-code URL. |
| Medium | References are inline only and the final practical discussion lacks an explicit closing label. | Add a short, connected closing section and a compact primary-source reference list; avoid duplicating a long bibliography unnecessarily. |
| Medium | The article's release check is dated November 14 while the C++ follow-up rechecked the release November 19. | Distinguish the recorded November 14 Python experiments from the November 19 release check. Keep exact OpenCV 5.1.0 and do not imply all CPU/GPU outputs are identical. |
| Publication | Canonical URL, author, first-publication timestamp, featured media URL, and newsletter Code URL are not yet live-page facts. | Fill these from actual publication state. Verify the released ZIP and checksum before wiring Code URL, and never submit the registration form during checks. |

## Assets and accessibility

All twelve referenced image assets exist and have descriptive, nonempty alt
text (53–119 characters). Their descriptive filenames and nearby explanations
are appropriate. No invented evidence or generic decorative image is needed.
The first measured photograph is 2085×728; keep a separate social thumbnail
rather than treating a narrow screenshot as a square crop.

The Markdown's original PNG/SVG references total **8,092,986 bytes**. The local
HTML already substitutes the retained WebP versions for photographs and plots,
bringing its referenced image payload to **1,735,030 bytes**, about **78.6% less**.
Preserve those compact equivalents on publication, with a working `src` fallback.
These are file-size measurements, not an HTTP transfer or speed-test result.
Google supports both [WebP and SVG and recommends useful alt text](https://developers.google.com/search/docs/appearance/google-images).
Intrinsic dimensions help reserve image space and reduce [layout shifts](https://web.dev/articles/optimize-cls).

The baseline HTML has one H1 and 25 unique heading IDs, with no duplicate IDs.
No table-of-contents fragment links are present at baseline. Keep a logical
heading hierarchy and verify the added contents list against rendered headings.

The requested featured thumbnail is now prepared at
`figures/robust-image-alignment-thumbnail.png` and `.webp`, **1600×900**, matching
the site's current 16:9 thumbnail convention. It uses the actual moving-photo
and aligned-overlay panels from `figures/graf_alignment.png`, plus readable
typography and an explanatory arrow. It contains no invented matches or
benchmark numbers. Full-size and 320×180 previews were visually inspected.
The PNG is 916,180 bytes; the WebP is 181,158 bytes. Source/crop provenance and
output SHA-256 values are retained in `seo-metadata.json`; the original composition
is retained in the published media. `scripts/make_thumbnail.py` now uses OpenCV
to regenerate the same layout with built-in Hershey typography; text rendering
therefore differs from the original thumbnail. Use the WebP
for the site where supported and retain the PNG master in the repository.

## Links, evidence, and accuracy

All sixteen external article links returned HTTP 200 in GET probes on November
19. Five are contextual internal LearnOpenCV links; the others are Oxford,
official OpenCV/LightGlue sources, and the ALIKED/LightGlue/MAGSAC papers. The
five existing internal links connect feature alignment, homographies, ECC,
neural feature matching, and document scanning. No extra unrelated link list
is needed for keywords. HTTP 200 confirms retrieval, not the correctness of
every cited statement; the version-specific API references and measured results
remain subject to the code audit.

No companion-code, LearnOpenCV GitHub, release ZIP, Dropbox, or raw-source URL
was found in the baseline article body. Keep those URLs in release/README
documentation and site metadata. The code gate is a newsletter distribution
flow, not a claim that public GitHub files are confidential or inaccessible.

The article is appropriately cautious about eight evaluation pairs from two
capture sequences, separate validation data, 25 timing trials on five common
success pairs, and synthetic versus photographed evidence. Keep that scope.
The stored Python GPU run reaches 8/8 LightGlue accuracy passes; the faithful
C++ CPU run reaches 7/8 because Graf6 has about 11.97 px independent error.
Where C++ availability is mentioned, keep the article's GPU results explicitly
scoped rather than implying backend-independent success or speed.

## Canonicalization and structured data

If the full article appears on both GitHub and LearnOpenCV, verify a self-
canonical LearnOpenCV URL on the owned website. GitHub Markdown does not give
the author control over the document head. Google may choose a canonical
among [duplicate pages](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls);
do not promise that a canonical directive can be set on GitHub or that ranking
will follow the publisher's preference.

Use the existing site SEO integration's Article/BlogPosting data once the real
author, public image URL, canonical URL, and first-publication timestamp are
known. Avoid a second conflicting schema block. Google's [Article guidance](https://developers.google.com/search/docs/appearance/structured-data/article)
recommends accurate authors, dates, headline, and representative images. Do not
invent a publication date or author URL for an unpublished draft.

Keep the visible FAQ because it helps readers, but do **not** add FAQPage markup
as a search-result tactic or promise FAQ rich results. Google's [May 8 and June
15, 2026 documentation updates](https://developers.google.com/search/updates)
state that FAQ rich results stopped appearing May 7 and the documentation was
removed in June.

## Final verification after edits

Before publishing, check exact H1/title, description/social metadata, one valid
canonical URL, all contents links and local assets, mobile image/table/code
legibility, exact version/results scope, and absence of code/download URLs in
the article body. After site publication, verify the actual page's metadata,
author and date schema, featured image, indexability, and newsletter popup's
verified Code URL. Do not claim a live performance score, search ranking, or
newsletter delivery test that was not measured. Rich Results Test and Search
Console URL Inspection are useful follow-ups once the real page exists.

## Prepublication review after the fixes

Reviewed the revised repository article/HTML and the WordPress handoff on
November 19, 2026. The requested title and 148-character description are
unchanged. The article now has concise learning outcomes, sixteen working
contents links, an explicit conclusion, and primary-source references. The
initial command enters the exact `Robust-Image-Alignment` extracted directory.
The release recheck is dated November 19 while the original Python experiments
remain clearly dated November 14. The relative raw-experiment hyperlink was
replaced with a plain-text reference to the evidence included in the download.

| Check | Repository preview | WordPress body fragment |
| --- | --- | --- |
| Unique heading IDs | 29 | 28 |
| H1 count | 1 | 0; the theme supplies the article title |
| Contents links / unresolved targets | 16 / 0 | 16 / 0 |
| Images with alt text and intrinsic dimensions | 12 / 12 | 12 / 12 |
| Missing referenced image files | 0 | 0 |
| Lazy images / high-priority first image | 11 / 1 | 11 / 1 |
| Malformed image tags | 0 | 0 |
| SyntaxHighlighter code blocks | Preview uses regular code blocks | 12 |
| Referenced image payload | 1,735,030 bytes | 1,841,694 bytes |

The revised prose is approximately 5,663 words excluding fenced code. There are
25 external-link occurrences pointing to the same sixteen unique destinations
already checked successfully, including five contextual LearnOpenCV articles.
The preview now contains the prepared meta description, matching Open Graph
title/description, article type, and a large-image Twitter card. Final canonical
and public social-image URLs intentionally remain unset until the real page and
media are created.

The GitHub source equation now uses `$$` display delimiters, consistent with
[GitHub's documented math syntax](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
The local preview preserves the same MathJax equation. The WordPress fragment
uses the equivalent readable expression `[u v w]ᵀ = H [x y 1]ᵀ`, followed by
`x′ = u/w, y′ = v/w`; its transpose and perspective division agree with the
source equation. No external math script is required for that WordPress form.

Both an image-tag closing issue found during review and the GitHub display-math
compatibility issue have been corrected. The prepared thumbnail remains
readable at 320×180. No further editorial/rendering publication blocker was
found in this review. Actual canonical/OG image URLs, author/date metadata,
indexability, and the site-managed newsletter button and GitHub landing link
still require verification against the published site; this local review does
not claim they have already been activated or tested.

## Production verification — November 19, 2026

Published post 80541 at https://learnopencv.com/robust-image-alignment/. Anonymous HTTP 200, the exact title and 148-character description, canonical, 1600×900 featured thumbnail, indexing directives, one H1, article schema dates, and article-body link policy all pass. The GitHub reader notice now links to this live article. All 13 uploaded media items have descriptive alt text. The public Download Example Code control opens the site-managed registration popup; First Name and Email are required and its Code URL equals the anonymously verified GitHub release ZIP. No form was submitted and actual email delivery was not tested.

The site emits existing AIOSEO/Schema Pro article metadata with matching title and dates; no extra FAQPage schema was added. Sitemap retrieval returned HTTP403 to the terminal, so sitemap inclusion was not claimed. No site-wide plugin, robots, security, or cache changes were made. Detailed live evidence is in ../publication/publication-verification.json.

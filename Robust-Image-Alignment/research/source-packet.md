# Robust Image Alignment source packet

Research checked on 2026-11-14. This is an editorial and implementation reference, not the article. Runtime assertions, environment manifests, executed notebook output, and experiment JSON remain the authority for what was actually tested.

## Critical implementation facts

1. Use the **released OpenCV 5.1.0 runtime**, the latest stable release found in the live GitHub release API. [Official release](https://github.com/opencv/opencv/releases/tag/5.1.0), [release API](https://api.github.com/repos/opencv/opencv/releases/latest), [5.1.0 release notes](https://github.com/opencv/opencv/wiki/OpenCV-Change-Logs#version510). The API reported `draft=false`, `prerelease=false`, and publication `2026-11-06T09:58:19Z`. Annotated tag `dd8d9c3833e6fbda076660cfbf2ae45611723481` peels to commit `e4f78fbcf1a773fbed69e14500e8b734963eb585`. Root is building from this release tag. PyPI still offers `opencv-python==5.0.0.93`, an older released runtime: do not describe that wheel as latest or as the tested 5.1 environment. The `.93` is a packaging revision; the wheel's runtime is 5.0.0.

2. Pin LightGlue's code. On this check `cvg/LightGlue` main resolved to **`5a650c444718caffc2669dc0094cde908f42dd73`**. The package reports `0.0`, so that package version does not uniquely identify the code. Its requirements declare `torch>=1.9.1`, `torchvision>=0.3`, NumPy, `opencv-python`, Matplotlib, and `kornia>=0.6.11`; there is no declared PyTorch upper bound. Pin a tested PyTorch/torchvision pair rather than presenting these broad minimums as a validated combination. [Requirements](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/requirements.txt), [package metadata](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/pyproject.toml).

3. The latest LightGlue commit adds an optional corner-origin keypoint convention, but the matcher still expects center-origin coordinates. Keep the default `origin="center"`, consistent with OpenCV. The extractor's `extract()` returns keypoints mapped back to the original input image; do not scale those points a second time. It supplies `image_size` in `(width, height)` order. [Pinned utilities](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/lightglue/utils.py#L132-L163).

4. The official reproducible ALIKED pairing is `ALIKED(model_name="aliked-n16", ...)` and `LightGlue(features="aliked", ...)`. This extractor outputs 128-dimensional descriptors, which the ALIKED LightGlue checkpoint expects. The `aliked-t16` variant outputs 64-dimensional descriptors. Descriptor dimension alone does not establish compatibility of other variants/checkpoints. [Pinned ALIKED configurations](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/lightglue/aliked.py#L613-L635), [matcher configurations](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/lightglue/lightglue.py#L351-L385).

5. OpenCV's `USAC_MAGSAC` option denotes **MAGSAC++ in the USAC framework**. In the 5.1.0 source preset it uses uniform sampling, MAGSAC scoring, and sigma-consensus local optimization. It does not select the Progressive-NAPSAC sampler also proposed in the MAGSAC++ paper. Explain the conceptual paper contribution separately from the exact OpenCV configuration. [OpenCV 5.1.0 preset](https://github.com/opencv/opencv/blob/5.1.0/modules/geometry/src/usac/ransac_solvers.cpp#L1176-L1183), [versioned USAC tutorial source](https://github.com/opencv/opencv/blob/5.1.0/doc/tutorials/calib3d/usac.markdown#L224).

6. Do not label the OpenCV call parameter-free. The passed reprojection threshold still influences returned inliers, stopping/verification, and quality calculations. The source maintains a separate maximum threshold: homography defaults to `7.5` in the model implementation, and initialization takes `max(passed_threshold, maximum_threshold)` for the marginalization bound. Thus it is inaccurate to describe `ransacReprojThreshold` simply as the one maximum sigma parameter. For homography, use pixel coordinates and explain the chosen pixel threshold and sensitivity. [5.1.0 threshold initialization](https://github.com/opencv/opencv/blob/5.1.0/modules/geometry/src/usac/ransac_solvers.cpp#L158-L164), [quality construction](https://github.com/opencv/opencv/blob/5.1.0/modules/geometry/src/usac/ransac_solvers.cpp#L314-L324), [default model parameters](https://github.com/opencv/opencv/blob/5.1.0/modules/geometry/src/usac/ransac_solvers.cpp#L1381-L1415). The [rendered USAC tutorial](https://docs.opencv.org/4.x/de/d3e/tutorial_usac.html) remains explanatory background; the pinned 5.1.0 source is stronger version-specific evidence.

7. `findTransformECC(reference, moving, warp, ...)` uses a warp from **reference to moving**, intended for `warpPerspective(moving, warp, reference_size, flags=INTER_LINEAR | WARP_INVERSE_MAP)`. If our normal alignment homography maps moving to reference, initialize ECC with its inverse and invert the refined result to preserve our public contract. Catch non-convergence and retain the accepted feature-based estimate. Only claim refinement improved results when measured independently. [OpenCV 5.1.0 ECC contract](https://github.com/opencv/opencv/blob/5.1.0/modules/video/include/opencv2/video/tracking.hpp#L319-L385), [implementation inverse-warp flags](https://github.com/opencv/opencv/blob/5.1.0/modules/video/src/ecc.cpp#L512-L519).

8. OpenCV 5 split the former `calib3d` module. The C++ source implementing these geometry routines lives in `modules/geometry`; Python's exposed root functions must be confirmed against the actual built bindings. Do not infer that a build necessarily includes new native LightGlue wrappers merely because current development documentation lists them. This tutorial's learned extractor/matcher can use the official PyTorch implementation while OpenCV 5 performs geometry and warping.

## Conceptual support and explanation boundaries

### ALIKED

Primary paper: [ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation](https://arxiv.org/abs/2304.03608), [readable full text](https://ar5iv.labs.arxiv.org/html/2304.03608). Primary code: [Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED), [pinned inference wrapper](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/lightglue/aliked.py).

The useful teaching point is sparse learned descriptor sampling: estimate supporting offsets around each detected keypoint, bilinearly gather feature values at those offsets, and combine them into a descriptor. A schematic can show fixed grid versus learned supporting positions, labelled as an explanatory illustration rather than actual predicted offsets. ALIKED still computes convolutional image feature maps; saying it avoids all dense image computation is wrong. Its descriptor head avoids building a descriptor at every pixel.

Optional diagram notation: supporting locations are `p + delta_m(p)`; a learned aggregation of sampled features produces `d(p)`. Define notation ourselves rather than copying the paper's equation layout. Do not claim guaranteed invariance to every geometric change. Report our measured behavior on rotations/viewpoints and texture-poor regions.

### LightGlue

Primary paper: [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643), [full text](https://ar5iv.labs.arxiv.org/html/2306.13643). Primary code: [pinned matcher](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/lightglue/lightglue.py).

Self-attention aggregates relationships within each feature set; cross-attention exchanges context between them. The lightweight assignment head combines pairwise similarity and unary matchability. The final correspondence filter enforces mutual maxima and a score threshold. Distinguish three quantities: whether a counterpart exists, how strong a candidate assignment is, and whether additional layers are expected to change the assignment. Pruning removes confidently unmatchable tokens; uncertain tokens are retained. Early exit is a pair-level decision based on sufficient confident assignments.

The positional encoding is applied to self-attention, not a geometric verification constraint between the two images. The matching network is not a replacement for a homography estimator or a guarantee that unrelated pairs will be rejected.

### MAGSAC and MAGSAC++

Original MAGSAC: [MAGSAC: marginalizing sample consensus](https://arxiv.org/abs/1803.07469). MAGSAC++: [MAGSAC++, a fast, reliable and accurate robust estimator](https://arxiv.org/abs/1912.05909). Author implementation: [danini/magsac](https://github.com/danini/magsac).

The original method marginalizes over a range of noise scales, avoiding commitment to one exact noise sigma for scoring. MAGSAC++ introduces a revised quality function and an efficient iteratively reweighted least-squares marginalization procedure. Progressive-NAPSAC is a separate sampling contribution of that paper. Keep this conceptual explanation distinct from OpenCV's full USAC preset. Comparing `cv2.RANSAC` with `cv2.USAC_MAGSAC` compares estimator configurations, not solely a replacement scoring function.

### Geometry and one-reference recognition

Primary explanatory sources: [OpenCV homography tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html), [feature matching and homography to find an object](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html). The first describes plane-induced and pure-rotation homographies. The second demonstrates locating a known planar target from a reference image and projecting its corners into a cluttered scene.

Define `H_moving_to_reference` explicitly. For aligned output, use `warpPerspective(moving, H_moving_to_reference, reference_size)` with normal transform semantics. For object localization, project reference corners using its inverse. HPatches ground truth generally maps sequence image 1 to image i; inspect each file and use the appropriate direction. Image-size order and matrix direction deserve teaching comments beside code.

Use recognizing a known visual appearance from one reference precisely. Pretrained models need no new target-specific training for this application. A book-cover match identifies a design/edition, not a unique physical copy, authenticity, or a general category learned from one example. A printed photograph of a target may match too. Textureless surfaces, unseen backsides, major deformation, small targets, repeated logos, and indistinguishable variants require different or additional evidence.

## Performance interpretation

[LightGlue's official benchmark](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/benchmark.py) extracts features before the timed matcher call. Its optional SuperGlue comparison is labelled SuperGlue-fast and sets five Sinkhorn iterations. Do not copy its FPS into a claim about end-to-end alignment or an unspecified default SuperGlue baseline. Timing needs warm-up, accelerator synchronization, repeated samples, and platform/configuration evidence. Cache the reference only in the experiment that actually measures cached-reference identification; pair alignment should count both image extractions.

Original experimental interpretation should dominate the article: count inliers as a diagnostic, use known geometry for actual alignment error, separate correct alignments from rejection decisions, and include missing-target queries. Similar pixel intensities or high ECC alone do not prove correct geometry, particularly on repeated patterns or low-texture images. Compare on a valid overlap mask; black warp borders and newly introduced content are not registration errors.

## Licences and provenance

[LightGlue's pinned README](https://github.com/cvg/LightGlue/blob/5a650c444718caffc2669dc0094cde908f42dd73/README.md#license) states Apache-2.0 for its code and LightGlue pretrained weights, with separate third-party terms. [ALIKED's licence](https://github.com/Shiaoming/ALIKED/blob/main/LICENSE) is BSD-3-Clause. Preserve attribution and licence notices when redistributing source. Do not generalize this statement to every feature extractor: original SuperPoint has separate restrictive terms, and Glue Factory supplies a distinct open implementation. Image/dataset rights need separate asset-by-asset evidence; model/code licences do not license photographs.

## Existing LearnOpenCV cross-references

These URLs returned HTTP 200 and their HTML canonical tags confirmed the exact URL on 2026-11-14:

| Canonical article | Suggested placement |
| --- | --- |
| [Feature Based Image Alignment using OpenCV](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/) | Classical baseline; explain that the new article extends the earlier form-registration example with learned correspondences and measured validation. |
| [Image Alignment ECC in OpenCV](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/) | Direct alignment/refinement and initialization discussion. |
| [Homography examples using OpenCV](https://learnopencv.com/homography-examples-using-opencv-python-c/) | Transform fundamentals and planarity/parallax boundary. |
| [Introduction to Feature Matching Using Neural Networks](https://learnopencv.com/feature-matching/) | Optional background on the broader matcher landscape. |
| [Automatic Document Scanner using OpenCV](https://learnopencv.com/automatic-document-scanner-using-opencv/) | Clarify document-boundary rectification versus registering a known document template. |

The first three articles carry July 31, 2026 technical corrections. Avoid describing them simply as obsolete 2015/2018 tutorials. Avoid the double-hyphen homography URL: it resolves, but the canonical URL uses one hyphen.

## Editorial and evidence checklist

- State the exact released OpenCV runtime prominently, with an executed version cell and manifest.
- Pin model code, model checksums, asset checksums, dependency versions, seeds, hardware, image dimensions, feature budgets, and estimator settings.
- Every article number must trace to retained result data or primary evidence. No anticipated speedup or accuracy gain should become a finding before experiments finish.
- Distinguish real captures, synthetically warped test cases, and illustrative diagrams in captions.
- Do not call a small selected image set a general benchmark or extrapolate rejection rates to deployment prevalence.
- Estimate thresholds on validation examples, then freeze them before evaluating held-out cases. Label exploratory selected demonstrations honestly if no independent tuning split exists.
- Report failures and unresolved cases; do not silently choose examples only because the preferred pipeline wins.
- Plot geometry error separately from number of matches and returned inliers.
- Use correctly directed homographies, valid warp masks, and distinct reference-support versus scene-support criteria for small-object localization.
- Show alignment images and inspect them at readable size; numerical metrics alone do not establish a compelling visual example.
- The complete notebook must execute top-to-bottom, access only bundled or checksum-verified downloaded data, and contain code that performs each advertised application.
- Article snippets must agree with the executed companion implementation; avoid explanatory pseudocode presented as a full runnable solution.
- Keep source summaries concise. Web sources generally have a 200-word paraphrase budget per source; alternate URLs for the same paper do not multiply that budget. Our original implementation explanations and experiments should be the article's main substance. Avoid quoting source prose.
- Keep the natural connected writing voice. Avoid formulaic hooks, generic lessons, hidden drafting placeholders, and unsupported superlatives.

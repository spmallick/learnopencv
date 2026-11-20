"""Populate the skill-scaffolded tutorial notebook with reproducible teaching cells.

The generated notebook is the reader-facing artifact. This small authoring
script keeps its code examples reviewable and makes editorial updates safe.
"""

from pathlib import Path
from textwrap import dedent

import nbformat


PROJECT = Path(__file__).resolve().parent
NOTEBOOK = PROJECT / "robust_image_alignment.ipynb"


def markdown(source):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(dedent(source).strip())


cells = [
    markdown(r"""
    # Robust image alignment with ALIKED LightGlue and MAGSAC++

    This companion notebook takes two photographs, finds corresponding details,
    estimates a homography, and checks whether the resulting alignment is usable.
    It also identifies a known planar appearance from **one reference image**.

    You should already know basic Python, NumPy arrays, and how to load images.
    The examples cover planar surfaces and approximately planar scenes. A single
    homography does not describe arbitrary 3D scenes with parallax or deformation.

    We will:
    1. Verify OpenCV **5.1.0**, the latest stable 5.x release checked on **November 14, 2026**, and the checksummed inputs.
    2. Extract ALIKED features, match with LightGlue, and fit geometry with MAGSAC++.
    3. Align real viewpoint, illumination, and document examples.
    4. Locate a known object design from one reference and reject absent targets.
    5. Measure independent geometric error, compare estimators, and test ECC refinement.
    6. Inspect a parallax failure and run a small, explicitly synthetic noise study.

    **Before running:** create the environment described in `README.md`, install
    the pinned dependencies, build OpenCV with `scripts/build_opencv.sh`, and select
    that environment's Jupyter kernel. Do not install another OpenCV wheel into the
    same environment: it would replace the source-built `cv2` namespace. The
    notebook checks the actual runtime rather than trusting a package filename.
    """),
    code(r"""
    import json
    import platform
    import subprocess
    import sys
    from io import BytesIO
    from pathlib import Path

    # Support launching Jupyter from this project or from the repository root.
    candidates = [Path.cwd(), Path.cwd() / "Robust-Image-Alignment"]
    PROJECT = next((path.resolve() for path in candidates
                    if (path / "robust_image_alignment.py").is_file()), None)
    if PROJECT is None:
        raise RuntimeError("Start Jupyter in Robust-Image-Alignment or its repository root.")
    sys.path.insert(0, str(PROJECT))

    import cv2
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from IPython.display import Image, Markdown, display

    # Use the released version for which the article records experimental evidence.
    assert cv2.__version__ == "5.1.0", f"Expected OpenCV 5.1.0; found {cv2.__version__} at {cv2.__file__}"
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    print({"OpenCV": cv2.__version__, "Python": platform.python_version(),
           "NumPy": np.__version__, "PyTorch": torch.__version__})
    print("cv2 loaded from:", cv2.__file__)
    """),
    markdown(r"""
    ## Download and verify the inputs and weights

    `download_assets.py` checks each file's SHA-256 digest and byte count against
    `assets_manifest.json`. Existing verified downloads are reused. The Oxford
    graffiti and Leuven sequences contain **real camera captures** and supplied
    reference-to-moving homographies; the later noise study is synthetic.

    The box and photographed-form examples are qualitative demonstrations with
    no independent ground-truth homography bundled. The manifest preserves
    provenance and licensing notes; downloading an image does not relicense it.
    Model weights are likewise pinned and verified before inference.
    """),
    code(r"""
    from download_assets import download_assets
    from robust_image_alignment import (
        AlignmentConfig, FeatureMatcher, MatchResult, draw_matches,
        estimate_alignment, image_corners, load_image, project_points,
        refine_ecc, warp_alignment,
    )
    from run_experiments import geometric_error

    ASSETS = download_assets(verbose=False)
    manifest = json.loads((PROJECT / "assets_manifest.json").read_text())
    print(f"Verified {len(manifest['files'])} input files.")
    verification = subprocess.run(
        [sys.executable, str(PROJECT / "download_models.py")],
        check=True, capture_output=True, text=True,
    )
    print(verification.stdout.strip().splitlines()[-1])
    pairs = {entry["id"]: entry for entry in manifest["pairs"]}
    """),
    markdown(r"""
    ## Look at the actual inputs

    The graffiti wall changes viewpoint; the Leuven building changes illumination.
    The form has perspective distortion and added handwriting. The box photograph
    contains a small known appearance within a cluttered scene. These are separate
    capture problems, not just rotations of one image.
    """),
    code(r"""
    def show_images(items, columns=2, width=12):
        # Render BGR or grayscale inputs without depending on a GUI backend.
        rows = (len(items) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(width, 3.7 * rows), squeeze=False)
        for axis, (title, image) in zip(axes.flat, items):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
            axis.imshow(rgb, cmap="gray" if image.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")
        for axis in list(axes.flat)[len(items):]:
            axis.axis("off")
        figure.tight_layout()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        display(Image(data=buffer.getvalue()))
        plt.close(figure)

    reference = load_image(ASSETS / pairs["graf_1_6"]["reference"])
    moving = load_image(ASSETS / pairs["graf_1_6"]["moving"])
    show_images([("Graffiti reference", reference), ("Real viewpoint change", moving)])
    """),
    markdown(r"""
    ## ALIKED finds the features and LightGlue matches them

    ALIKED's sparse deformable descriptor head learns where to sample supporting
    features around a keypoint. LightGlue then uses context within and between
    images to establish correspondences. Their pretrained weights form a pair:
    this notebook uses **`aliked-n16`** and **`LightGlue(features="aliked")`**.

    The wrapper converts OpenCV BGR uint8 images to RGB tensors in `[0, 1]`, caps
    the longest side at 1024 pixels without upsampling, and retains the extractor's
    original-image coordinates. Do **not** scale those coordinates a second time.
    """),
    code(r"""
    matcher = FeatureMatcher("lightglue", device="auto", max_keypoints=2048, resize=1024)
    print("Inference device:", matcher.device)
    # Model loading is outside timing; references can be extracted once and cached.
    reference_features = matcher.extract(reference)
    moving_features = matcher.extract(moving)
    print("Keypoints:", reference_features.keypoints.shape, moving_features.keypoints.shape)
    print("Descriptors:", reference_features.descriptors.shape, moving_features.descriptors.shape)
    print("Batched tensors:", {
        name: tuple(value.shape) for name, value in reference_features.tensors.items()
        if hasattr(value, "shape")
    })
    """),
    markdown(r"""
    ## Follow the tensor indices back to image coordinates

    Each row of LightGlue's `matches` contains one index into the reference
    keypoints and one into the moving keypoints. The batched tensors keep one
    image pair in batch dimension 0. This cell makes that indexing explicit;
    `matcher.match(reference_features, moving_features)` provides the reusable
    equivalent with timing and consistent empty-input handling.

    LightGlue can stop early on confident pairs and prune confidently unmatchable
    points. Its match scores are not probabilities of object identity.
    """),
    code(r"""
    with torch.inference_mode():
        raw_matches = matcher.matcher({
            "image0": reference_features.tensors,
            "image1": moving_features.tensors,
        })
    # Detach only the small correspondence output needed by OpenCV geometry.
    indices = raw_matches["matches"][0].detach().cpu().numpy()
    points_reference = reference_features.keypoints[indices[:, 0]]
    points_moving = moving_features.keypoints[indices[:, 1]]
    matches = matcher.match(reference_features, moving_features)
    print(f"{len(points_reference)} tentative matches; coordinate shape {points_reference.shape}.")
    show_images([("Tentative matches (up to 70 shown)", draw_matches(reference, moving, matches, max_draw=70))], columns=1)
    """),
    markdown(r"""
    ## MAGSAC++ estimates the homography

    Incorrect correspondences remain possible even with a learned matcher.
    OpenCV's `USAC_MAGSAC` configuration uses MAGSAC++ to estimate robust geometry.
    It accounts for uncertainty in the noise scale, while the OpenCV call still
    takes an error threshold and iteration/confidence settings.

    **Direction matters:** source points come first, so the call below estimates
    **moving → reference**. The threshold is in original reference-image pixels.
    A homography models a plane, or a camera rotation about its projection center;
    it does not make parallax disappear.
    """),
    code(r"""
    H_direct, inlier_mask = cv2.findHomography(
        points_moving, points_reference,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.999,
    )
    if H_direct is None:
        raise RuntimeError("No homography could be estimated.")
    print("Moving-to-reference homography:\n", H_direct / H_direct[2, 2])
    print("Estimator-selected inliers:", int(inlier_mask.sum()))

    # The production helper also checks spatial support and projected geometry.
    config = AlignmentConfig()
    result = estimate_alignment(matches, reference.shape, moving.shape, config, "magsac")
    assert result.accepted, result.reasons
    print({"accepted": result.accepted, "inliers": result.num_inliers,
           "reference_coverage": round(result.reference_coverage, 3)})
    """),
    markdown(r"""
    ## Warp the image and preserve a valid-pixel mask

    `warpPerspective` uses the moving-to-reference homography directly. Missing
    pixels can appear black, but real image content can also be black; therefore
    validity comes from a warped source mask, not from testing pixel intensity.
    The overlay below retains the reference wherever source support is absent.
    """),
    code(r"""
    aligned, valid_mask = warp_alignment(moving, result, reference.shape)
    overlay = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0)
    overlay[valid_mask == 0] = reference[valid_mask == 0]
    show_images([
        ("Reference", reference), ("Aligned moving image", aligned),
        ("Valid source support", valid_mask), ("50/50 overlay", overlay),
    ])
    """),
    markdown(r"""
    ## Measure on independent ground truth

    Inlier reprojection error measures how well a model fits the matches it
    selected. A wrong model can fit its own selected matches very well. Here we
    instead construct a regular grid in the reference, map it with the dataset's
    supplied homography, and evaluate estimated alignment only inside the true
    overlap. Errors are in original reference-image pixels.

    These are small teaching experiments on two capture sequences. The split
    holds out image pairs, not entirely new scenes, so it is not a generalization
    benchmark or a claim of state-of-the-art performance.
    """),
    code(r"""
    H_ground_truth = np.asarray(pairs["graf_1_6"]["homography_matrix"]).reshape(3, 3)
    grid_error = geometric_error(result.H, H_ground_truth, reference.shape, moving.shape)
    print({key: round(value, 3) if isinstance(value, float) else value
           for key, value in grid_error.items()})
    print("Fit residual on selected inliers:", round(result.median_reprojection_error, 3), "px")
    # Retain the actual images and estimate for the later refinement experiment.
    demonstrations = {"graf_1_6": (reference, moving, matches, result, H_ground_truth)}
    """),
    markdown(r"""
    ## Repeat on illumination change and a photographed document

    The same pipeline handles different kinds of appearance change without
    retraining on either target. The form's filled fields genuinely differ from
    the blank template; an alignment overlay is not an instruction to erase or
    ignore those differences. Neither form input supplies independent ground
    truth, so its result is a qualitative demonstration.
    """),
    code(r"""
    for name, reference_path, moving_path in [
        ("leuven_1_6", pairs["leuven_1_6"]["reference"], pairs["leuven_1_6"]["moving"]),
        ("document", "learnopencv/form.jpg", "learnopencv/scanned-form.jpg"),
    ]:
        ref, mov = load_image(ASSETS / reference_path), load_image(ASSETS / moving_path)
        found = matcher.match_images(ref, mov)
        fitted = estimate_alignment(found, ref.shape, mov.shape, config)
        print(name, {"accepted": fitted.accepted, "inliers": fitted.num_inliers, "reasons": fitted.reasons})
        assert fitted.accepted, (name, fitted.reasons)
        warped, support = warp_alignment(mov, fitted, ref.shape)
        mixed = cv2.addWeighted(ref, 0.5, warped, 0.5, 0)
        mixed[support == 0] = ref[support == 0]
        show_images([(name + " reference", ref), (name + " input", mov),
                     (name + " aligned", warped), (name + " overlay", mixed)])
        truth = np.asarray(pairs[name]["homography_matrix"]).reshape(3, 3) if name in pairs else None
        demonstrations[name] = (ref, mov, found, fitted, truth)
        if truth is not None:
            print("Independent grid error:", geometric_error(fitted.H, truth, ref.shape, mov.shape))
    """),
    markdown(r"""
    ## Identify and locate a known appearance from one reference

    We now enroll one image of the box design and look for it in a cluttered
    photograph. Cache the reference features, match a query image, check geometry,
    and project the reference boundary into the scene with the **inverse** H.

    This identifies a known appearance, such as a particular cover or label. It
    does not learn an entire semantic category from one example, distinguish
    physical copies with identical printing, or recognize an unseen backside.
    The acceptance gates below are transparent heuristics, not a calibrated
    probability or a proven false-acceptance operating point.
    """),
    code(r"""
    target = load_image(ASSETS / "opencv/box.png")
    scene = load_image(ASSETS / "opencv/box_in_scene.png")
    enrolled = matcher.extract(target)  # One reference image; no target-specific training.
    query_features = matcher.extract(scene)
    identification_matches = matcher.match(enrolled, query_features)
    identification = estimate_alignment(identification_matches, target.shape, scene.shape, config)
    assert identification.accepted, identification.reasons

    localized = scene.copy()
    # The helper has checked that the projected boundary is finite and plausible.
    quad = np.rint(identification.projected_reference_quad).astype(np.int32)
    cv2.polylines(localized, [quad], True, (30, 230, 30), 3, cv2.LINE_AA)
    print({"identified": identification.accepted, "inliers": identification.num_inliers,
           "reference_coverage": round(identification.reference_coverage, 3)})
    show_images([("The single reference", target), ("Reference located in scene", localized)])
    """),
    markdown(r"""
    ## An absent target should produce a rejection

    Repeat the same decision on two unrelated images. Passing these examples is
    a useful regression check, not a measured false-positive rate. A production
    catalogue needs many hard negatives, including similar designs and repeated
    patterns. A small target should not be rejected merely for occupying little
    of the scene: spatial coverage is checked in the reference coordinates.
    """),
    code(r"""
    for negative_path in ["opencv/fruits.jpg", "opencv/building.jpg"]:
        negative = load_image(ASSETS / negative_path)
        found = matcher.match(enrolled, matcher.extract(negative))
        decision = estimate_alignment(found, target.shape, negative.shape, config)
        print(Path(negative_path).name, {"accepted": decision.accepted,
              "matches": len(found.reference_points), "reasons": decision.reasons})
        assert not decision.accepted, "Unexpected acceptance: inspect this negative example."
    """),
    markdown(r"""
    ## Extend to a small reference catalogue

    Candidate verification scales naturally to a few enrolled references. For a
    large catalogue, first retrieve a short candidate list, then run LightGlue
    and geometric verification on that list. If several candidates pass, preserve
    the ambiguity instead of silently calling the largest match count an identity.
    """),
    code(r"""
    catalogue = {
        "box design": (target, enrolled),
        "graffiti wall": (reference, reference_features),
        "blank form": (demonstrations["document"][0], matcher.extract(demonstrations["document"][0])),
    }
    candidates = []
    for name, (reference_image, cached_features) in catalogue.items():
        found = matcher.match(cached_features, query_features)
        candidate = estimate_alignment(found, reference_image.shape, scene.shape, config)
        candidates.append((name, candidate.accepted, candidate.num_inliers))
    print("Candidate decisions (name, accepted, inliers):", candidates)
    accepted_names = [name for name, accepted, _ in candidates if accepted]
    print("Identification:", accepted_names[0] if len(accepted_names) == 1 else "ambiguous or absent")
    """),
    markdown(r"""
    ## Hold the matches fixed when comparing estimators

    This comparison uses identical LightGlue correspondences and the same pixel
    threshold for OpenCV's RANSAC and `USAC_MAGSAC` configurations. Their internal
    robust-estimation machinery differs; this is a comparison of complete OpenCV
    configurations, not an isolated replacement of one scoring formula.

    Match count and fitted inlier residual are diagnostics. Independent geometric
    error is the accuracy measure when dataset ground truth is available.
    """),
    code(r"""
    estimator_rows = []
    for estimator in ["ransac", "magsac"]:
        fitted = estimate_alignment(matches, reference.shape, moving.shape, config, estimator)
        errors = geometric_error(fitted.H, H_ground_truth, reference.shape, moving.shape)
        estimator_rows.append({"estimator": estimator, "accepted": fitted.accepted,
                               "inliers": fitted.num_inliers,
                               "independent_median_px": errors["grid_median_error_px"],
                               "fit_median_px": fitted.median_reprojection_error})
    for row in estimator_rows:
        print(row)
    """),
    markdown(r"""
    ## Test ECC refinement rather than assuming it helps

    ECC directly optimizes image agreement. It requires a reasonable initial
    alignment and may improve, worsen, or fail on a pair. OpenCV ECC expects a
    reference-to-moving initialization; `refine_ecc` inverts the module's H before
    optimization and inverts the returned matrix afterward. It also masks the
    initial overlap to reduce clutter and invalid-border influence.

    Compare before and after against independent geometry. A high correlation
    value alone is not proof of a better alignment.
    """),
    code(r"""
    for name in ["graf_1_6", "leuven_1_6"]:
        ref, mov, found, fitted, truth = demonstrations[name]
        before = geometric_error(fitted.H, truth, ref.shape, mov.shape)["grid_median_error_px"]
        try:
            refined_H, correlation = refine_ecc(ref, mov, fitted.H)
            after = geometric_error(refined_H, truth, ref.shape, mov.shape)["grid_median_error_px"]
            print(name, {"before_px": round(before, 3), "after_px": round(after, 3),
                         "ECC_correlation": round(correlation, 4), "improved": after < before})
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            print(name, "ECC failed:", str(error).splitlines()[0])
    """),
    markdown(r"""
    ## A single homography cannot remove parallax

    These stereo photographs contain surfaces at different depths. A homography
    may align one dominant surface while leaving other surfaces doubled. Our
    geometric acceptance gates can accept a well-supported plane; they do not
    establish that every pixel in a 3D scene is aligned. Inspect the overlay and
    use a different motion model or local/dense method when the task requires it.
    """),
    code(r"""
    aloe_reference = load_image(ASSETS / "opencv/aloeL.jpg")
    aloe_moving = load_image(ASSETS / "opencv/aloeR.jpg")
    aloe_matches = matcher.match_images(aloe_reference, aloe_moving)
    aloe_result = estimate_alignment(aloe_matches, aloe_reference.shape, aloe_moving.shape, config)
    print({"accepted_plane": aloe_result.accepted, "inliers": aloe_result.num_inliers,
           "reasons": aloe_result.reasons})
    if aloe_result.H is not None:
        # Explicitly permit diagnostic warping even if the acceptance gate rejected it.
        aloe_warped, aloe_valid = warp_alignment(aloe_moving, aloe_result, aloe_reference.shape,
                                                 require_accepted=False)
        aloe_overlay = cv2.addWeighted(aloe_reference, 0.5, aloe_warped, 0.5, 0)
        aloe_overlay[aloe_valid == 0] = aloe_reference[aloe_valid == 0]
        show_images([("Real stereo reference", aloe_reference), ("Homography overlay: inspect different depths", aloe_overlay)])
    """),
    markdown(r"""
    ## A controlled synthetic noise experiment

    This study generates point correspondences from a known homography, adds
    Gaussian coordinate noise, and replaces exactly 60% with uniform outliers.
    Both estimators receive identical correspondences for each seed. It isolates
    a useful failure mechanism but does not model repeated structures, coherent
    false matches, image formation, or domain shift.

    The short notebook run uses three seeds. The saved article experiment records
    its larger seed count separately; do not confuse either with real captures.
    """),
    code(r"""
    from run_experiments import run_estimator_study

    synthetic_rows = run_estimator_study(seeds=3)
    for noise in [0.5, 1.5, 3.0]:
        for estimator in ["ransac", "magsac"]:
            selected = [row for row in synthetic_rows
                        if row["noise_sigma_px"] == noise and row["threshold_px"] == 3.0
                        and row["estimator"] == estimator]
            errors = [row["grid_median_error_px"] for row in selected
                      if row["grid_median_error_px"] is not None]
            print({"synthetic_sigma_px": noise, "estimator": estimator,
                   "accepted_runs": sum(row["accepted"] for row in selected),
                   "runs": len(selected), "finite_fits": len(errors),
                   "median_finite_model_error_px": round(float(np.median(errors)), 3) if errors else None})
    """),
    markdown(r"""
    ## Read the article's stored evidence or rerun the complete benchmark

    Stored results are evidence from the recorded machine, **not timings measured
    by this notebook kernel**. The benchmark compares SIFT + ratio matching,
    ALIKED + ratio matching, and ALIKED + LightGlue, each with RANSAC and MAGSAC++.
    A result passes its stated teaching criterion when accepted and below 3 px
    median independent overlap-grid error. Accuracy counts use all eight
    evaluation pairs. For comparable processing times, use only the same five
    pairs that all three MAGSAC++ pipelines successfully aligned in the stored
    article experiment, with five repetitions per pair: 25 trials per pipeline.
    The cell derives and prints that common-success intersection; a local rerun
    may produce a different intersection.

    Each timed common-success result includes both image extractions, matching,
    robust estimation and acceptance checks, and image/mask warping. It excludes
    disk I/O, model loading, warm-up, and visualization. Rejected runs, which skip
    warping, are excluded from this timing comparison.

    Pass count is the main accuracy summary. The median finite-model error is a
    secondary diagnostic over estimated models with finite errors, including
    rejected models; `finite_fits` makes missing estimates explicit. It is not
    an all-pairs accuracy average and must be read alongside the pass count.

    Set `RUN_FULL_BENCHMARK = True` to create a separate local evidence file.
    Reference/moving feature extraction is included in the uncached pair timing;
    repeated identification with cached references is a different workload.
    """),
    code(r"""
    RUN_FULL_BENCHMARK = False
    evidence_path = PROJECT / "results" / "experiments.json"
    if RUN_FULL_BENCHMARK:
        evidence_path = PROJECT / "results" / "notebook-rerun.json"
        subprocess.run([
            sys.executable, str(PROJECT / "run_experiments.py"),
            "--device", matcher.device, "--repeats", "3", "--study-seeds", "12",
            "--output", str(evidence_path),
        ], check=True)
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
        print("Stored experiment environment:", evidence["environment"])
        print("Configuration:", evidence["protocol"])
        methods = ["sift", "aliked-nn", "lightglue"]
        evaluation = [row for row in evidence["real_pairs"]
                      if row["split"] == "evaluation" and row["estimator"] == "magsac"]
        # Use the first repetition to select pair IDs, not the fastest measured run.
        common_ids = set.intersection(*[
            {row["id"] for row in evaluation
             if row["method"] == method and row["repeat"] == 0 and row["success_3px"]}
            for method in methods
        ])
        print("Common-success timing pairs:", sorted(common_ids))
        for method in methods:
            selected = [row for row in evaluation if row["method"] == method]
            # Accuracy uses every evaluation pair; timing uses every common-pair repeat.
            once = [row for row in selected if row["repeat"] == 0]
            errors = [row["grid_median_error_px"] for row in once
                      if row["grid_median_error_px"] is not None]
            timings = [row["pipeline_ms"] for row in selected if row["id"] in common_ids]
            percentiles = np.percentile(timings, [10, 90]) if timings else None
            print({"method": method, "passes": sum(row["success_3px"] for row in once),
                   "pairs": len(once), "finite_fits": len(errors),
                   "median_finite_model_error_px": round(float(np.median(errors)), 3) if errors else None,
                   "common_success_timing_pairs": len(common_ids), "timing_trials": len(timings),
                   "median_common_success_processing_ms": round(float(np.median(timings)), 2) if timings else None,
                   "p10_p90_common_success_processing_ms": (
                       [round(float(value), 2) for value in percentiles] if percentiles is not None else None)})
    else:
        print("No stored benchmark file here. Set RUN_FULL_BENCHMARK=True to generate local evidence.")
    """),
    markdown(r"""
    ## Exercise: reject a fit supported by one small patch

    A large match count can hide poor spatial support. Before running the answer
    cell, predict what happens if we keep only matches from the top-left quarter
    of the reference. Which acceptance gate should catch the problem? Would the
    same scene-coverage rule be appropriate for a small box inside a large scene?

    Change the retained region or `min_reference_coverage`, rerun the cell, and
    inspect the decision. Threshold selection belongs on validation examples.
    """),
    code(r"""
    # Runnable answer scaffold: select a small spatial patch without changing matches.
    keep = ((matches.reference_points[:, 0] < reference.shape[1] / 4)
            & (matches.reference_points[:, 1] < reference.shape[0] / 4))
    patch_matches = MatchResult(matches.reference_points[keep], matches.moving_points[keep])
    strict_support = AlignmentConfig(min_reference_coverage=0.10)
    patch_result = estimate_alignment(patch_matches, reference.shape, moving.shape, strict_support)
    print({"retained_matches": int(keep.sum()), "accepted": patch_result.accepted,
           "coverage": patch_result.reference_coverage, "reasons": patch_result.reasons})
    assert not patch_result.accepted
    """),
    markdown(r"""
    ## What to carry into your own application

    Preserve coordinate conventions, cache reference features when appropriate,
    check spatial evidence and projected geometry, and distinguish a usable
    planar alignment from a valid model for every scene pixel. Match scores,
    inlier counts, and ECC correlation each describe part of the evidence.

    For a deployment, test hard negatives and held-out capture conditions, measure
    full application latency, and choose rejection thresholds for that workload.
    Small-texture targets, severe blur, repeated designs, parallax, and deformation
    remain important cases. An optional extension is candidate retrieval before
    LightGlue verification for a larger reference catalogue.

    Related LearnOpenCV tutorials:
    - [Feature Based Image Alignment using OpenCV](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
    - [Image Alignment using ECC](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
    - [Homography Examples using OpenCV](https://learnopencv.com/homography-examples-using-opencv-python-c/)
    - [Introduction to Feature Matching Using Neural Networks](https://learnopencv.com/feature-matching/)

    Primary references:
    [ALIKED](https://arxiv.org/abs/2304.03608),
    [LightGlue](https://arxiv.org/abs/2306.13643),
    [MAGSAC++](https://arxiv.org/abs/1912.05909),
    [OpenCV 5.1.0 release](https://github.com/opencv/opencv/releases/tag/5.1.0),
    [Oxford affine dataset](https://www.robots.ox.ac.uk/~vgg/research/affine/).
    """),
]


def main():
    # Retain the notebook scaffold's metadata while replacing its example cells.
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells = cells
    notebook.metadata.kernelspec = {"display_name": "Python 3 (OpenCV 5.1)", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"}
    notebook.metadata["tutorial"] = {"opencv_release": "5.1.0", "executed_evidence": "results/experiments.json"}
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Wrote {len(cells)} cells to {NOTEBOOK}")


if __name__ == "__main__":
    main()

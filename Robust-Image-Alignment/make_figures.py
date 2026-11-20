"""Create the article diagrams and figures from actual alignment computations.

The diagrams-only route needs NumPy, Matplotlib and OpenCV. The full route also
uses the companion alignment module and its selected neural dependencies. Every
photographic result comes from that module; schematic points are confined to
explicitly labelled explanatory diagrams.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Rendering must also work on headless experiment hosts.

import matplotlib.pyplot as plt
from matplotlib._tight_bbox import adjust_bbox
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Circle, ConnectionPatch, FancyArrowPatch, FancyBboxPatch
import numpy as np
import cv2


PROJECT_DIR = Path(__file__).resolve().parent
NAVY = "#18283F"
BLUE = "#087EA4"
TEAL = "#008A7A"
AMBER = "#C87513"
RED = "#BA3744"
GRAY = "#64748B"
PALE = "#EEF3F8"


def configure_style() -> None:
    """Use the same readable typography and restrained palette in every figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": NAVY,
            "axes.titlecolor": NAVY,
            "axes.labelcolor": NAVY,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def figure_bgr(fig: plt.Figure, padding: float) -> np.ndarray:
    """Render the original tight crop at 150 dpi without an intermediate image file."""
    original_dpi = fig.dpi
    original_layout = fig.get_layout_engine()
    restore_bbox = None
    try:
        fig.set_dpi(150)
        fig.canvas.draw()
        # Reuse Matplotlib's savefig crop adjustment, including artists outside the
        # initial canvas, instead of clipping a previously rendered full-size image.
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(padding)
        restore_bbox = adjust_bbox(fig, bounds, fixed_dpi=150)
        fig.set_layout_engine("none")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    finally:
        if restore_bbox is not None:
            restore_bbox()
        fig.set_layout_engine(original_layout)
        fig.set_dpi(original_dpi)
    # Agg supplies RGB channel order; OpenCV's encoders expect BGR. Composite
    # transparent pixels on white so publication backgrounds stay consistent.
    if np.any(rgba[..., 3] != 255):
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = np.rint(rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def save_figure(fig: plt.Figure, path: Path, *, vector: bool = False) -> None:
    """Encode the rendered raster with OpenCV and optionally retain editable SVG text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if vector:
            fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.18)
            # Matplotlib adds trailing spaces to SVG path lines; keep repository diffs clean.
            svg = path.with_suffix(".svg")
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        image = figure_bgr(fig, 0.18 if vector else 0.16)
        png = path.with_suffix(".png")
        if not cv2.imwrite(str(png), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"Cannot write figure image: {png}")
        webp = path.with_suffix(".webp")
        if not cv2.imwrite(str(webp), image, [cv2.IMWRITE_WEBP_QUALITY, 91]):
            raise OSError(f"Cannot write figure image: {webp}")
    finally:
        plt.close(fig)


def diagram_box(ax: plt.Axes, x: float, y: float, width: float, height: float,
                title: str, detail: str, color: str) -> None:
    """Draw one labelled stage; text remains vector text in the SVG export."""
    ax.add_patch(
        FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.025",
                       facecolor=PALE, edgecolor=color, linewidth=1.7)
    )
    ax.text(x + width / 2, y + height * 0.65, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.29, detail, ha="center", va="center", fontsize=10)


def make_pipeline_diagram(output_dir: Path) -> None:
    """Show the distinct responsibilities of extraction, matching and geometry."""
    fig, ax = plt.subplots(figsize=(13.8, 3.9))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.95, "From two photographs to a checked alignment", fontsize=20,
            fontweight="bold", va="top")
    labels = [
        ("ALIKED", "Keypoints +\nlocal descriptors", BLUE),
        ("LightGlue", "Tentative\ncorrespondences", BLUE),
        ("MAGSAC++", "Robust homography\nestimation", TEAL),
        ("Check geometry", "If accepted, warp image\nand source mask", TEAL),
    ]
    starts = [0.02, 0.27, 0.52, 0.77]
    for x, (title, detail, color) in zip(starts, labels):
        diagram_box(ax, x, 0.32, 0.21, 0.43, title, detail, color)
    for x in starts[:-1]:
        ax.add_patch(FancyArrowPatch((x + 0.215, 0.535), (x + 0.245, 0.535),
                                    arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=GRAY))
    ax.text(0.02, 0.12,
            "The homography maps the moving image into reference coordinates.\n"
            "Return a rejection if the checks fail. Even an accepted model can misalign objects at other depths.",
            fontsize=11, color=GRAY, va="center")
    save_figure(fig, output_dir / "pipeline", vector=True)


def make_aliked_diagram(output_dir: Path) -> None:
    """Explain deformable descriptor support without depicting invented model output."""
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.7))
    fixed = np.array([[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]])
    learned = np.array([[-1.7, -0.8], [-0.7, -0.55], [0.5, -0.95], [-1.2, 0.2],
                        [0.15, 0.3], [1.2, -0.2], [-0.8, 1.1], [0.6, 1.35], [1.55, 0.85]])
    for ax, title, points, color in zip(
        axes, ["A fixed sampling grid", "Learned sampling offsets"], [fixed, learned], [GRAY, BLUE]
    ):
        ax.set(xlim=(-2.3, 2.3), ylim=(-2.1, 2.1), aspect="equal")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xticks(np.arange(-2, 2.1, 0.5))
        ax.set_yticks(np.arange(-2, 2.1, 0.5))
        ax.grid(color="#DAE3EC", linewidth=0.7)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # A bent edge is only a conceptual feature-map structure for the diagram.
        ax.plot([-2.2, -0.9, 0.6, 2.2], [-1.5, -0.4, 0.2, 1.6], color="#CBD5E1", linewidth=15,
                solid_capstyle="round", zorder=1)
        for point in points:
            ax.plot([0, point[0]], [0, point[1]], color=color, alpha=0.45, linewidth=1, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], s=75, c=color, edgecolors="white", linewidths=1.2, zorder=3)
        ax.add_patch(Circle((0, 0), 0.105, facecolor=AMBER, edgecolor="white", linewidth=1, zorder=4))
    fig.suptitle("ALIKED learns where to gather descriptor support", fontsize=18, fontweight="bold", y=1.02)
    fig.text(0.5, -0.015,
             "Orange marks the keypoint. Nine sample positions are drawn for clarity. The aliked-n16 model uses 16.",
             ha="center", fontsize=10, color=GRAY)
    fig.tight_layout()
    save_figure(fig, output_dir / "aliked_sampling", vector=True)


def make_lightglue_diagram(output_dir: Path) -> None:
    """Illustrate adaptive depth while clearly separating schematic and measured data."""
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.97, "LightGlue can stop before its final layer", fontsize=19, fontweight="bold", va="top")
    for y, label, active in [(0.60, "Earlier confident assignments", 3), (0.30, "More refinement required", 9)]:
        ax.text(0.02, y + 0.015, label, fontsize=11, va="center")
        for layer in range(9):
            x = 0.36 + layer * 0.065
            enabled = layer < active
            ax.add_patch(FancyBboxPatch((x, y - 0.055), 0.047, 0.14,
                                       boxstyle="round,pad=0.008,rounding_size=0.01",
                                       facecolor=BLUE if enabled else PALE,
                                       edgecolor=BLUE if enabled else "#CFD9E3", linewidth=1))
            ax.text(x + 0.0235, y + 0.015, str(layer + 1), color="white" if enabled else GRAY,
                    ha="center", va="center", fontsize=10)
        if active == 3:
            ax.text(0.515, y + 0.16, "example: stop after layer 3", color=TEAL, fontsize=10, ha="center")
    ax.text(0.02, 0.08, "Conceptual example only; the actual stopping layer depends on the image pair and configuration.",
            fontsize=10, color=GRAY)
    save_figure(fig, output_dir / "lightglue_depth", vector=True)


def show_image(ax: plt.Axes, image_bgr: np.ndarray, title: str) -> None:
    """Maintain image aspect ratio and leave room for concise panel headings."""
    ax.imshow(image_bgr[..., ::-1])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")


def result_summary(result: Any) -> str:
    """Report observed evidence without treating confidence as an identity probability."""
    status = "Accepted" if result.accepted else "Rejected"
    text = f"{status} · {result.num_matches} tentative matches · {result.num_inliers} inliers"
    if result.median_reprojection_error is not None:
        text += f" · median inlier residual {result.median_reprojection_error:.2f} px"
    return text


def masked_overlay(reference: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Blend only where the warped moving image has genuine source support."""
    overlay = reference.copy()
    supported = valid > 0
    blend = np.rint(0.5 * reference.astype(np.float32) + 0.5 * aligned.astype(np.float32)).astype(np.uint8)
    overlay[supported] = blend[supported]
    return overlay


def render_alignment_strip(core: Any, reference: np.ndarray, moving: np.ndarray, result: Any,
                           title: str, path: Path, *, diagnostic: bool = False) -> None:
    """Show inputs, masked overlay and mask rather than a misleading border blend."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.7), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    show_image(axes[0], reference, "Reference")
    show_image(axes[1], moving, "Moving photograph")
    if result.H is not None and (result.accepted or diagnostic):
        aligned, valid = core.warp_alignment(moving, result, reference.shape, require_accepted=not diagnostic)
        show_image(axes[2], masked_overlay(reference, aligned, valid),
                   "Diagnostic 50/50 overlay" if diagnostic else "Aligned 50/50 overlay")
        # Teal support remains visible even when the mask fills the whole frame.
        support_colors = ListedColormap(["#DEE5ED", TEAL])
        axes[3].imshow(valid > 0, cmap=support_colors, vmin=0, vmax=1)
        axes[3].set_title("Valid source support", fontsize=11, fontweight="bold", pad=10)
        axes[3].axis("off")
        axes[3].text(0.5, -0.055, "Teal: valid · gray: outside", ha="center", va="top",
                     fontsize=9, color=GRAY, transform=axes[3].transAxes)
    else:
        for ax in axes[2:]:
            ax.axis("off")
        axes[2].text(0.5, 0.5, "No accepted\nalignment", ha="center", va="center", color=RED, fontsize=15,
                     transform=axes[2].transAxes)
    fig.suptitle(title, fontsize=19, fontweight="bold", y=0.99)
    footer = result_summary(result)
    if diagnostic:
        footer = footer.replace("Accepted", "Geometry checks passed", 1)
        footer += "\nA single homography can fit one depth region while other depths remain misaligned."
    elif result.reasons:
        footer += "\n" + " ".join(result.reasons[:2])
    fig.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=10, color=GRAY)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.84, bottom=0.16, wspace=0.05)
    save_figure(fig, path)


def render_match_figure(reference: np.ndarray, moving: np.ndarray, matches: Any, result: Any,
                        path: Path, title: str, max_draw: int = 60) -> None:
    """Plot only actual correspondences, using a reproducible subset for readability."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    groups = [(np.arange(len(matches.reference_points)), AMBER, "Tentative correspondences"),
              (np.flatnonzero(result.inliers), TEAL, "MAGSAC++ geometric inliers")]
    for row, (indices, color, label) in enumerate(groups):
        for ax, image in zip(axes[row], [reference, moving]):
            show_image(ax, image, "Reference" if ax is axes[row, 0] else "Moving photograph")
        selected = indices[np.linspace(0, len(indices) - 1, min(max_draw, len(indices)), dtype=int)] if len(indices) else []
        for idx in selected:
            p0, p1 = matches.reference_points[idx], matches.moving_points[idx]
            for ax, point in zip(axes[row], [p0, p1]):
                ax.scatter(point[0], point[1], s=10, c=color, linewidths=0, zorder=3)
            line = ConnectionPatch(p0, p1, "data", "data", axesA=axes[row, 0], axesB=axes[row, 1],
                                   color=color, linewidth=0.6, alpha=0.65, zorder=2)
            fig.add_artist(line)
        axes[row, 0].text(0, 1.14, f"{label}: showing {len(selected)} of {len(indices)}", color=color,
                          fontsize=12, fontweight="bold", transform=axes[row, 0].transAxes)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "Displayed lines are a subset; the estimator uses every supplied correspondence.",
             ha="center", fontsize=10, color=GRAY)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.87, bottom=0.055, hspace=0.35, wspace=0.03)
    save_figure(fig, path)


def render_object_figure(core: Any, reference: np.ndarray, moving: np.ndarray, result: Any, path: Path) -> None:
    """Project the reference boundary only from the computed homography."""
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.3), gridspec_kw={"width_ratios": [0.95, 1.5, 0.95]})
    show_image(axes[0], reference, "One reference image")
    show_image(axes[1], moving, "Scene + estimated reference boundary")
    if result.projected_reference_quad is not None:
        quad = np.vstack((result.projected_reference_quad, result.projected_reference_quad[0]))
        axes[1].plot(quad[:, 0], quad[:, 1], color=TEAL if result.accepted else RED, linewidth=2.8)
    if result.accepted:
        rectified, valid = core.warp_alignment(moving, result, reference.shape)
        show_image(axes[2], rectified, "Rectified matched surface")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "Identification rejected", color=RED, ha="center", transform=axes[2].transAxes)
    fig.suptitle("Locate a known printed design from one reference", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.025, result_summary(result) + "\nThis identifies the visible design, not a unique physical package.",
             ha="center", fontsize=10, color=GRAY)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.84, bottom=0.15, wspace=0.05)
    save_figure(fig, path)


def render_rejection(reference: np.ndarray, moving: np.ndarray, result: Any, path: Path) -> None:
    """Expose the actual target-absent decision, including an unexpected acceptance."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    show_image(axes[0], reference, "Known reference")
    show_image(axes[1], moving, "Target-absent photograph")
    axes[2].axis("off")
    color = RED if result.accepted else TEAL
    label = "False acceptance" if result.accepted else "Rejected"
    axes[2].text(0.05, 0.68, label, fontsize=21, fontweight="bold", color=color, transform=axes[2].transAxes)
    axes[2].text(0.05, 0.43, f"{result.num_matches} tentative matches\n{result.num_inliers} geometric inliers",
                 fontsize=12, linespacing=1.8, transform=axes[2].transAxes)
    fig.suptitle("A useful pipeline also handles an absent target", fontsize=18, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "A single easy negative illustrates the behavior; it does not establish a false-acceptance rate.",
             ha="center", fontsize=10, color=GRAY)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.12, wspace=0.08)
    save_figure(fig, path)


def make_measured_benchmark_figure(document: dict[str, Any], output_dir: Path) -> None:
    """Plot the small real-pair study, retaining rejected and unestimated models.

    Each cell is the median over repeated estimates of that pair's median
    ground-truth overlap-grid error. Repeats do not increase the number of
    distinct evaluated image pairs. A capped color scale avoids allowing an
    implausible homography to erase differences among successful estimates.
    """
    rows = [row for row in document.get("real_pairs", []) if row.get("split") == "evaluation"]
    if not rows:
        return
    pair_ids = sorted({row["id"] for row in rows})
    configurations = [
        ("sift", "ransac", "SIFT + RANSAC"),
        ("sift", "magsac", "SIFT + MAGSAC++"),
        ("aliked-nn", "ransac", "ALIKED + NN + RANSAC"),
        ("aliked-nn", "magsac", "ALIKED + NN + MAGSAC++"),
        ("lightglue", "ransac", "ALIKED + LightGlue + RANSAC"),
        ("lightglue", "magsac", "ALIKED + LightGlue + MAGSAC++"),
    ]
    cap = 10.0
    values = np.full((len(configurations), len(pair_ids)), np.nan)
    cells: list[dict[str, Any]] = []
    for row_index, (method, estimator, label) in enumerate(configurations):
        for column, pair_id in enumerate(pair_ids):
            group = [row for row in rows if row["id"] == pair_id
                     and row["method"] == method and row["estimator"] == estimator]
            errors = [float(row["grid_median_error_px"]) for row in group
                      if row.get("grid_median_error_px") is not None
                      and np.isfinite(row["grid_median_error_px"])]
            median = float(np.median(errors)) if errors else None
            accepted_count = sum(bool(row.get("accepted")) for row in group)
            if median is not None:
                values[row_index, column] = min(cap, median)
            cells.append({"row": row_index, "column": column, "id": pair_id, "method": method,
                          "estimator": estimator, "median_grid_error_px": median,
                          "accepted_runs": accepted_count, "runs": len(group)})
    fig, ax = plt.subplots(figsize=(14.0, 6.8))
    palette = LinearSegmentedColormap.from_list("alignment_error", ["#EDF8F6", "#82CEC1", "#F3D49D", "#CE6B59"])
    palette.set_bad("#DADEE3")  # A missing fit has no numerical error.
    image = ax.imshow(values, cmap=palette, vmin=0, vmax=cap, aspect="auto")
    labels = [pair_id.replace("graf_", "Graffiti ").replace("leuven_", "Leuven ").replace("_", "→")
              for pair_id in pair_ids]
    ax.set_xticks(np.arange(len(pair_ids)), labels, fontsize=10)
    ax.set_yticks(np.arange(len(configurations)), [item[2] for item in configurations], fontsize=10)
    ax.tick_params(length=0, pad=12)
    ax.set_xticks(np.arange(-0.5, len(pair_ids), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(configurations), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for cell in cells:
        median, count, total = cell["median_grid_error_px"], cell["accepted_runs"], cell["runs"]
        if median is None:
            text = "no H"
        else:
            text = f"{median:.2f}"
        if median is not None and total and count == 0:
            text += "\n× rejected"
        elif median is not None and count != total:
            text += f"\n{count}/{total} accepted"
        ax.text(cell["column"], cell["row"], text, ha="center", va="center", fontsize=10,
                color="#5F1E28" if count != total else NAVY,
                fontweight="bold" if median is None or median > cap else "normal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
    colorbar.set_ticks([0, 3, 5, 10])
    colorbar.set_label("Median grid error (original reference pixels)", fontsize=10)
    fig.suptitle("Alignment error on eight held-out pairs from two real sequences", fontsize=18,
                 fontweight="bold", y=0.98)
    repeats = document.get("protocol", {}).get("repeats", "multiple")
    device = document["environment"]["device"].upper()
    fig.text(0.5, 0.90, f"Neural inference on {device} · median over {repeats} runs per pair · smaller is better",
             ha="center", fontsize=11, color=GRAY)
    fig.text(0.5, 0.055,
             "Colors saturate at 10 px. Cell labels give the measured error. × marks a rejected estimate.\n"
             "Gray means no homography was estimated. Pair labels identify the reference and moving frames.",
             ha="center", fontsize=10, color=GRAY, linespacing=1.5)
    fig.subplots_adjust(left=0.265, right=0.92, top=0.84, bottom=0.16)
    save_figure(fig, output_dir / "benchmark_grid_errors")
    (output_dir / "benchmark_plot_data.json").write_text(
        json.dumps({"color_cap_px": cap, "evaluation_pair_ids": pair_ids, "cells": cells},
                   indent=2, allow_nan=False) + "\n"
    )


def make_threshold_sensitivity_figure(document: dict[str, Any], output_dir: Path) -> None:
    """Show measured synthetic noise sensitivity and the separate gate decisions."""
    rows = document.get("estimator_study", [])
    if not rows:
        return
    sigmas = sorted({float(row["noise_sigma_px"]) for row in rows})
    thresholds = sorted({float(row["threshold_px"]) for row in rows})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(13.5, 6.4), sharey=True, squeeze=False)
    max_upper = 0.0
    for ax, sigma in zip(axes[0], sigmas):
        table_rows = []
        for estimator, color, label in [("ransac", AMBER, "RANSAC"),
                                        ("magsac", BLUE, "MAGSAC++")]:
            medians, lower, upper, annotations = [], [], [], []
            for threshold in thresholds:
                group = [row for row in rows if float(row["noise_sigma_px"]) == sigma
                         and float(row["threshold_px"]) == threshold and row["estimator"] == estimator]
                errors = [float(row["grid_median_error_px"]) for row in group
                          if row.get("grid_median_error_px") is not None
                          and np.isfinite(row["grid_median_error_px"])]
                medians.append(float(np.median(errors)) if errors else np.nan)
                lower.append(float(np.percentile(errors, 25)) if errors else np.nan)
                upper.append(float(np.percentile(errors, 75)) if errors else np.nan)
                annotations.append(f"{sum(bool(row.get('accepted')) for row in group)}/{len(group)}")
            max_upper = max(max_upper, max((value for value in upper if np.isfinite(value)), default=0.0))
            ax.plot(thresholds, medians, marker="o", color=color, linewidth=2, label=label)
            ax.fill_between(thresholds, lower, upper, color=color, alpha=0.12, linewidth=0)
            table_rows.append([label, *annotations])
        ax.set_title(f"Synthetic point noise σ = {sigma:g} px", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(thresholds)
        ax.set_xlabel("Shared threshold (px)", fontsize=10, labelpad=10)
        ax.grid(axis="y", color="#DEE5ED", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=10)
        # A separate compact table keeps acceptance counts legible when the
        # two error curves overlap, without placing labels across the zero axis.
        ax.text(0.5, -0.31, "Geometric-gate acceptances", ha="center", fontsize=10,
                color=GRAY, transform=ax.transAxes)
        table = ax.table(cellText=table_rows, colLabels=["Threshold", *[f"{value:g} px" for value in thresholds]],
                         cellLoc="center", bbox=[0, -0.75, 1.0, 0.37])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row_index, column), cell in table.get_celld().items():
            cell.set_edgecolor("#DEE5ED")
            cell.set_linewidth(0.7)
            if row_index == 0:
                cell.set_facecolor(PALE)
            elif column == 0:
                cell.get_text().set_color(AMBER if row_index == 1 else BLUE)
                cell.get_text().set_fontweight("bold")
    for ax in axes[0]:
        ax.set_ylim(0, max(4.7, max_upper * 1.2))
    axes[0, 0].set_ylabel("Median ground-truth grid error (px)", fontsize=10)
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=10)
    fractions = sorted({float(row["outlier_fraction"]) for row in rows})
    outlier_label = f"{fractions[0]:.0%} outliers" if len(fractions) == 1 else "controlled outlier fractions"
    seeds = len({row["seed"] for row in rows})
    fig.suptitle("MAGSAC++ still needs a sensible threshold", fontsize=18, fontweight="bold", y=1.00)
    fig.text(0.5, 0.02,
             f"Synthetic correspondences only · {outlier_label} · {seeds} paired random seeds per setting\n"
             "The reprojection threshold and median inlier residual limit change together. Shading shows the interquartile range.",
             ha="center", fontsize=10, color=GRAY, linespacing=1.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.40, wspace=0.18)
    save_figure(fig, output_dir / "magsac_threshold_sensitivity")


def rerender_saved_showcases(args: argparse.Namespace) -> None:
    """Re-render actual saved homographies without re-running neural inference.

    This route intentionally cannot reconstruct match-line plots: those need
    their original point coordinates. It refreshes only image/overlay figures
    and retains all original matching and estimation evidence in the metadata.
    """
    import cv2
    import robust_image_alignment as core
    from download_assets import download_assets

    metadata = json.loads(args.reuse_metadata.read_text())
    assets = download_assets(args.assets_dir, verbose=False)
    titles = {
        "graf_alignment": "Real viewpoint change: graffiti sequence 1 to 6",
        "leuven_alignment": "Real illumination change: Leuven sequence 1 to 6",
        "document_alignment": "Align a photographed form with its blank template",
        "parallax_failure": "Parallax: one transform cannot align every depth",
    }
    for row in metadata["showcases"]:
        reference = core.load_image(assets / row["reference"])
        moving = core.load_image(assets / row["moving"])
        # These are recorded scalar counts, not fabricated correspondence masks.
        result = SimpleNamespace(
            H=None if row["H"] is None else np.asarray(row["H"], np.float64),
            accepted=row["accepted"], reasons=row["reasons"], num_matches=row["num_matches"],
            num_inliers=row["num_inliers"], median_reprojection_error=row["median_reprojection_error_px"],
            projected_reference_quad=(None if row["projected_reference_quad"] is None
                                      else np.asarray(row["projected_reference_quad"], np.float32)),
        )
        name = row["id"]
        if name == "object_identification":
            render_object_figure(core, reference, moving, result, args.output_dir / name)
        elif name == "target_absent":
            render_rejection(reference, moving, result, args.output_dir / name)
        elif name in titles:
            render_alignment_strip(core, reference, moving, result, titles[name], args.output_dir / name,
                                   diagnostic=name == "parallax_failure")
    metadata["rendering_opencv_version"] = cv2.__version__
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def make_showcases(args: argparse.Namespace) -> None:
    """Run every photographic example through the shared, tested implementation."""
    import cv2
    import robust_image_alignment as core
    from download_assets import download_assets

    assets = download_assets(args.assets_dir, verbose=False)
    matcher = core.FeatureMatcher("lightglue", args.device, args.max_keypoints, args.resize)
    config = core.AlignmentConfig()
    examples = [
        ("graf_alignment", "oxford/graf/img1.png", "oxford/graf/img6.png",
         "Real viewpoint change: graffiti sequence 1 to 6"),
        ("leuven_alignment", "oxford/leuven/img1.png", "oxford/leuven/img6.png",
         "Real illumination change: Leuven sequence 1 to 6"),
        ("document_alignment", "learnopencv/form.jpg", "learnopencv/scanned-form.jpg",
         "Align a photographed form with its blank template"),
        ("object_identification", "opencv/box.png", "opencv/box_in_scene.png", "Known-design localization"),
        ("parallax_failure", "opencv/aloeL.jpg", "opencv/aloeR.jpg", "Parallax: one transform cannot align every depth"),
        ("target_absent", "opencv/box.png", "opencv/fruits.jpg", "Target-absent rejection"),
    ]
    reports = []
    cache: dict[str, Any] = {}
    for name, ref_name, moving_name, title in examples:
        reference, moving = core.load_image(assets / ref_name), core.load_image(assets / moving_name)
        if ref_name not in cache:
            cache[ref_name] = matcher.extract(reference)
        if moving_name not in cache:
            cache[moving_name] = matcher.extract(moving)
        matches = matcher.match(cache[ref_name], cache[moving_name])
        result = core.estimate_alignment(matches, reference.shape, moving.shape, config, "magsac")
        if name == "object_identification":
            render_object_figure(core, reference, moving, result, args.output_dir / name)
            render_match_figure(reference, moving, matches, result, args.output_dir / "object_matches",
                                "A learned match still needs geometric support")
        elif name == "target_absent":
            render_rejection(reference, moving, result, args.output_dir / name)
        else:
            render_alignment_strip(core, reference, moving, result, title, args.output_dir / name,
                                   diagnostic=name == "parallax_failure")
            if name == "graf_alignment":
                render_match_figure(reference, moving, matches, result, args.output_dir / "match_filtering",
                                    "Viewpoint change: tentative matches and geometric inliers")
        report = result.to_dict()
        report.update({"id": name, "reference": ref_name, "moving": moving_name,
                       "extraction_ms": matches.extraction_ms, "matching_ms": matches.matching_ms})
        reports.append(report)
        print(f"{name}: {result_summary(result)}", flush=True)
    metadata = {
        "opencv_version": cv2.__version__, "method": "ALIKED-n16 + LightGlue",
        "device": matcher.device, "max_keypoints": args.max_keypoints, "resize": args.resize,
        "configuration": asdict(config), "showcases": reports,
        "note": "Actual computed outputs. Schematic diagram sampling locations and exit layers are illustrative only.",
    }
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Keep diagram rendering available before the neural runtime is installed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "figures")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--diagrams-only", action="store_true")
    parser.add_argument("--results", type=Path, help="Measured experiments JSON; otherwise use the project results if present.")
    parser.add_argument("--results-only", action="store_true", help="Render diagrams and measured plots without running models.")
    parser.add_argument("--reuse-metadata", type=Path,
                        help="Re-render saved actual showcase homographies without neural inference; match plots remain unchanged.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    make_pipeline_diagram(arguments.output_dir)
    make_aliked_diagram(arguments.output_dir)
    make_lightglue_diagram(arguments.output_dir)
    if arguments.diagrams_only:
        return
    results_path = arguments.results or PROJECT_DIR / "results" / "experiments.json"
    if results_path.is_file():
        document = json.loads(results_path.read_text())
        make_measured_benchmark_figure(document, arguments.output_dir)
        make_threshold_sensitivity_figure(document, arguments.output_dir)
    elif arguments.results is not None:
        raise FileNotFoundError(results_path)
    if arguments.results_only:
        return
    if arguments.reuse_metadata:
        rerender_saved_showcases(arguments)
    else:
        make_showcases(arguments)


if __name__ == "__main__":
    main()

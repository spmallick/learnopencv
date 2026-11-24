"""Redraw the article's batch-one latency chart from per-session medians."""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")  # Readers can regenerate the figure on a headless benchmark host.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
MODELS = {"vit_b_16": "ViT-B/16", "resnet18": "ResNet-18"}
BACKENDS = {"opencv-cpu": "OpenCV CPU", "opencv-cuda": "OpenCV CUDA", "ort-cuda": "ORT CUDA"}
PHASES = {
    "forward": ("Warmed forward", "#007593"),
    "end_to_end": ("PNG file to label", "#D67C26"),
}


def plot(sessions, output, hardware):
    with sessions.open(encoding="utf-8", newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["variant"] == "fixed" and row["batch"] == "1"]
    # Make each session count once even if its p50 differs substantially from another session's.
    samples = {(model, backend, phase):
               [float(r[f"{phase}_p50_ms"]) for r in rows if r["model"] == model and r["backend"] == backend]
               for model in MODELS for backend in BACKENDS for phase in PHASES}
    if not all(samples.values()):
        raise ValueError("Expected batch-one fixed-shape sessions for both models and all three backends.")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.15))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.69, bottom=0.24, wspace=0.17)
    ink = "#172B3A"
    fig.text(0.033, 0.94, "Time to classify one image", color=ink, weight="bold", size=23)
    fig.text(0.035, 0.87, "Fixed 1 × 3 × 224 × 224 input   ·   FP32   ·   TF32 off", color=ink, size=12)
    fig.legend(handles=[Patch(color=c, label=label) for label, c in PHASES.values()],
               loc="center right", bbox_to_anchor=(0.965, 0.87), ncol=2, frameon=False,
               handlelength=1.1, handletextpad=0.5, columnspacing=1.25, fontsize=11)

    for ax, (model, title) in zip(axes, MODELS.items()):
        # Each panel starts at zero and has its own scale; the text gives the absolute latency.
        limit = max(max(samples[model, b, p]) for b in BACKENDS for p in PHASES)
        ax.set_xlim(0, limit * 1.25)
        ax.set_ylim(-0.55, 2.55)
        ax.set_title(title, loc="left", pad=9, color=ink, size=16, fontweight="bold")
        for y, backend in zip([2, 1, 0], BACKENDS):
            for shift, (phase, (_, color)) in zip([0.17, -0.17], PHASES.items()):
                values = samples[model, backend, phase]
                center, low, high = median(values), min(values), max(values)
                ax.barh(y + shift, center, height=0.27, color=color, zorder=2)
                # Capped whiskers show the min/max session medians. They are not confidence bounds.
                ax.errorbar(center, y + shift, xerr=[[center - low], [high - center]],
                            fmt="none", ecolor=ink, elinewidth=1.3, capsize=3, capthick=1.1, zorder=3)
                ax.text(high + 0.016 * limit, y + shift, f"{center:.1f}", color=ink,
                        va="center", fontsize=10)
        ax.set_yticks([2, 1, 0], list(BACKENDS.values()) if ax is axes[0] else [])
        ax.tick_params(axis="y", length=0, labelsize=11, pad=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x", color="#DCE3E7", linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", length=0, labelsize=10, pad=4)
        ax.set_xlabel("Latency (ms)", color=ink, labelpad=7)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.spines["left"].set_color("#8597A0")

    fig.text(0.035, 0.09, "Bars show the median of session medians. Whiskers span their range.",
             color=ink, fontsize=10.5)
    if hardware:
        fig.text(0.035, 0.045, hardware, color="#536675", fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=HERE.parent / "measurements/sessions.csv")
    parser.add_argument("--output", type=Path, default=HERE / "fixed-batch-latency.png")
    parser.add_argument("--hardware", default="Intel Core i7-6850K (6 threads used)   ·   NVIDIA GeForce GTX 1080 Ti")
    args = parser.parse_args()
    plot(args.sessions, args.output, args.hardware)

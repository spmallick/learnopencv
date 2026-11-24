"""Compare one-image ViT requests with the preceding batch held fixed or changed."""

import argparse
import csv
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def one_image_probe(probes, mode, previous_batch):
    # Skip startup. Count a forward only if the CSV actually shows the requested
    # transition into B=1; B is the number of images in the input tensor.
    sequence = sorted((r for r in probes if r["model"] == "vit_b_16" and r["shape_mode"] == mode),
                      key=lambda r: int(r["forward"]))
    rows = [current for previous, current in zip(sequence, sequence[1:])
            if int(current["forward"]) == int(previous["forward"]) + 1
            and int(previous["batch"]) == previous_batch and current["batch"] == "1"]
    if not rows:
        raise ValueError(f"No measured {previous_batch} to 1 transition in the {mode} CUDA probe")
    return rows


def plot(directory, output):
    sessions = read_csv(directory / "sessions.csv")
    probes = read_csv(directory / "cuda-api-calls.csv")
    latency, volume, launches, allocations, requests = [], [], [], [], []
    for variant, mode, previous in [("dynamic-held-1", "held", 1), ("dynamic-changing", "changing", 2)]:
        rows = [r for r in sessions if r["model"] == "vit_b_16" and r["backend"] == "opencv-cuda"
                and r["variant"] == variant and r["batch"] == "1"]
        if len(rows) != 1:
            raise ValueError(f"Expected one latency session with one-image observations for {variant}")
        latency.append(float(rows[0]["forward_p50_ms"]))
        requests.append(int(rows[0]["forward_samples"]))

        observed = one_image_probe(probes, mode, previous)
        volume.append(median(int(r["h2d_bytes"]) for r in observed) / 2**20)
        launches.append(median(int(r["runtime_kernel_launches"]) for r in observed))
        allocations.append(median(int(r["malloc_calls"]) for r in observed))

    ink, muted = "#172B3A", "#536675"
    colors = ["#87A7B5", "#007593"]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 4.65))
    fig.subplots_adjust(left=0.05, right=0.97, top=0.61, bottom=0.245, wspace=0.3)
    fig.text(0.035, 0.915, "One-image ViT requests after a batch change", color=ink, weight="bold", size=20)
    fig.text(0.035, 0.835, "ViT-B/16  ·  Symbolic batch size  ·  OpenCV 5.1 CUDA  ·  GeForce GTX 1080 Ti",
             color=muted, size=11.5)
    fig.legend(handles=[Patch(color=c, label=label) for c, label in zip(colors, (
        "After one image (batch size held at one)", "After two images (alternating one and two)"))],
        bbox_to_anchor=(0.028, 0.795), loc="upper left", ncol=2, frameon=False,
        handlelength=1, handletextpad=0.5, columnspacing=1.6, fontsize=11)
    metrics = [
        ("Forward latency\n(ms)", latency, ".2f"),
        ("Host-to-GPU data\ncopied (MiB)", volume, ".3f"),
        ("Runtime kernel\nlaunches", launches, "g"),
        ("Device allocations\n(cudaMalloc calls)", allocations, "g"),
    ]
    for ax, (label, values, digits) in zip(axes, metrics):
        # Each metric has its own units and zero baseline. Label every measured value.
        high = max(values) or 1
        ax.bar([0, 1], values, color=colors, width=0.52)
        for x, value in enumerate(values):
            ax.text(x, value + high * 0.045, format(value, digits), ha="center", va="bottom",
                    fontsize=12, color=ink, fontweight="semibold")
        ax.set_ylim(0, high * 1.26)
        ax.set_xlim(-0.6, 1.6)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel(label, fontsize=11, weight="semibold", color=ink, labelpad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#BAC8D0")

    fig.text(0.035, 0.095, f"Latency is the median of {requests[0]} held-size and {requests[1]} alternating one-image forwards.",
             color=muted, fontsize=10.1)
    fig.text(0.035, 0.048, "Copy volume and calls are per-forward observations from separate CUPTI runs on a second GTX 1080 Ti.",
             color=muted, fontsize=10.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=HERE.parent / "measurements")
    parser.add_argument("--output", type=Path, default=HERE / "previous-batch-cost.png")
    args = parser.parse_args()
    plot(args.measurements, args.output)

"""Compare the two bundled classifiers with fresh processes and identical inputs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BACKENDS = ("opencv-cpu", "opencv-cuda", "ort-cuda")
MODELS = ("vit_b_16", "resnet18")
HERE = Path(__file__).resolve().parent


def cases(repeats):
    # Rotate backend order for the independent fixed-input sessions.
    for trial in range(repeats):
        start = trial % len(BACKENDS)
        for model in MODELS[::1 if trial % 2 == 0 else -1]:
            for backend in BACKENDS[start:] + BACKENDS[:start]:
                yield model, backend, "fixed", trial + 1

    # Comparing two schedules on the *same* symbolic file separates dynamic
    # metadata from actually changing the size of requests between forwards.
    for model in MODELS:
        for backend in BACKENDS:
            yield model, backend, "dynamic-held-1", 1
            yield model, backend, "dynamic-held-2", 1
            yield model, backend, "dynamic-changing", 1
        for backend in ("opencv-cuda", "ort-cuda"):
            yield model, backend, "fixed-tf32", 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=HERE / "assets")
    parser.add_argument("--output", type=Path, default=HERE / "runs")
    parser.add_argument("--repeats", type=int, default=3, help="fresh sessions per fixed baseline")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if min(args.repeats, args.runs, args.threads, args.warmup) <= 0:
        parser.error("repeats, runs, threads and warmup must be positive")

    assets, output = args.assets.resolve(), args.output.resolve()
    # Different run sets get separate directories so a failed case cannot inherit old results.
    if output.exists() and any(output.iterdir()):
        parser.error(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = [
        dict(case=f"{model}_{backend}_{variant}_{trial}", model=model, backend=backend,
             variant=variant, trial=trial, status=None)
        for model, backend, variant, trial in cases(args.repeats)
    ]
    manifest = output / "matrix.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        model, backend, variant = record["model"], record["backend"], record["variant"]
        graph = "dynamic" if variant.startswith("dynamic") else "fixed"
        batches = {"dynamic-held-2": "2", "dynamic-changing": "1,2"}.get(variant, "1")
        name = record["case"]
        destination = output / name
        command = [
            sys.executable, str(HERE / "benchmark.py"),
            "--model", str(assets / f"{model}-{graph}.onnx"),
            "--image", str(assets / "chelsea.png"), "--image", str(assets / "coffee.png"),
            "--labels", str(assets / "labels.txt"), "--backend", backend,
            "--batches", batches,
            "--tf32", "on" if variant == "fixed-tf32" else "off",
            "--runs", str(args.runs), "--warmup", str(args.warmup),
            "--threads", str(args.threads), "--output", str(destination),
        ]
        print(name, flush=True)
        with (output / f"{name}.log").open("w") as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        record["status"] = status
        # Retain useful results if a model fails placement or an external run is interrupted.
        manifest.write_text(json.dumps(records, indent=2) + "\n")
        if status:
            print(f"  failed (see {name}.log)", flush=True)
    return int(any(row["status"] for row in records))


if __name__ == "__main__":
    sys.exit(main())

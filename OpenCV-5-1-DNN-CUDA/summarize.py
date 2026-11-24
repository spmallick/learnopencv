"""Summarize timing samples and compare full logits against the fixed OpenCV CPU run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def outputs(folder, result):
    # Batch two can contain the same photos in a different order from batch one.
    # Associate each score vector with its actual photograph before comparing.
    with np.load(folder / "logits.npz", allow_pickle=False) as archive:
        for case in result["predictions"]:
            scores = archive[case["key"]]
            if scores.shape != (len(case["images"]), 1000) or not np.isfinite(scores).all():
                raise ValueError(f"Invalid ImageNet logits in {folder}")
            for name, row in zip(case["images"], scores):
                yield case["batch"], name, row


def summarize(runs, destination):
    matrix = read_json(runs / "matrix.json")
    failed = [record["case"] for record in matrix if record["status"] != 0]
    # Keep failed cells visible. Avoid generating plausible tables for an incomplete experiment.
    if failed:
        raise ValueError("Review failed or unfinished cells in matrix.json: " + ", ".join(failed))

    references, hashes = {}, {}
    for record in matrix:
        if (record["backend"], record["variant"], record["trial"]) == ("opencv-cpu", "fixed", 1):
            folder = runs / record["case"]
            references[record["model"]] = {name: row for _, name, row in outputs(folder, read_json(folder / "result.json"))}

    summaries, samples = [], []
    for record in matrix:
        folder = runs / record["case"]
        result = read_json(folder / "result.json")
        filename, digest = result["model"], result["model_sha256"]
        if hashes.setdefault(filename, digest) != digest:
            raise ValueError(f"The ONNX file changed between runs: {filename}")
        by_batch = {}
        for batch, name, logits in outputs(folder, result):
            reference = references[record["model"]][name]
            top = np.argsort(logits)[-5:]
            ref_top = np.argsort(reference)[-5:]
            by_batch.setdefault(batch, []).append((
                int(top[-1] == ref_top[-1]), len(set(top) & set(ref_top)),
                float(np.max(np.abs(logits - reference))),
            ))
        for batch, agreement in by_batch.items():
            row = dict(case=record["case"], model=record["model"], backend=record["backend"],
                       variant=record["variant"], trial=record["trial"], batch=batch)
            # Cold timing always belongs to the first scheduled batch in this session.
            row["cold_batch"] = result["batches"][0]
            row.update(result["cold"] if batch == row["cold_batch"]
                       else dict.fromkeys(result["cold"]))
            for phase in ("forward", "end_to_end"):
                # Report B=1 and B=2 separately, including when they alternate within a session.
                data = result[phase]
                values = [t for b, t in zip(data["batches"], data["samples_ms"]) if b == batch]
                if not values:
                    raise ValueError(f"{folder}: too few measured requests for batch {batch}")
                row[phase + "_samples"] = len(values)
                row[phase + "_p50_ms"] = float(np.median(values))
                row[phase + "_p95_ms"] = float(np.percentile(values, 95))
            row["gpu_after_warm_mib"] = result["memory"]["gpu_after_warm_mib"]
            row["peak_rss_mib"] = result["memory"]["peak_rss_mib"]
            row["predictions_checked"] = len(agreement)
            row["top1_matches"] = sum(a[0] for a in agreement)
            row["min_top5_overlap"] = min(a[1] for a in agreement)
            row["max_abs_logit_error"] = max(a[2] for a in agreement)
            placement = result["placement"]
            row["cuda_ops"] = placement.get("cuda_ops", "")
            row["cpu_ops_in_gpu_run"] = placement.get("cpu_ops", "")
            row["cpu_op_types_in_gpu_run"] = json.dumps(placement.get("cpu_op_types", {}), sort_keys=True)
            row["sha256"] = digest
            summaries.append(row)
        for phase in ("forward", "end_to_end"):
            data = result[phase]
            for i, (batch, ms) in enumerate(zip(data["batches"], data["samples_ms"])):
                samples.append(dict(case=record["case"], phase=phase, index=i, batch=batch, ms=ms))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "sessions.csv", summaries)
    write_csv(destination / "samples.csv", samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs, args.output)

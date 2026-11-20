"""Validate native C++ results against the original Python implementation.

Python is only the independent validation harness. Each subprocess executes the
native application directly; the C++ program does not start or import Python.
Shared correspondences isolate geometry before complete feature pipelines are
compared on real captured images. Failures retain their measured diagnostics.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from download_assets import download_assets
from robust_image_alignment import (
    AlignmentConfig, FeatureMatcher, MatchResult, estimate_alignment,
    load_image, project_points, warp_alignment,
)
from run_experiments import geometric_error


def require(condition, message):
    """Use explicit checks so optimized Python execution cannot disable the gates."""
    if not condition:
        raise RuntimeError(message)


def read_matrix(value):
    """Accept nested report arrays and OpenCV FileStorage's matrix representation."""
    if value is None:
        return None
    if isinstance(value, dict) and "data" in value:
        return np.asarray(value["data"], np.float64).reshape(value["rows"], value["cols"])
    return np.asarray(value, np.float64)


def maximum_difference(left, right):
    """Report the largest absolute difference while handling empty arrays explicitly."""
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape, f"Array shapes differ: {left.shape} versus {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0


def validate_outputs(directory, accepted, dumped_features):
    expected = {"report.json", "matches.jpg"}
    if accepted:
        expected |= {"aligned.png", "valid_mask.png", "overlay.jpg", "reference_boundary.jpg"}
    if dumped_features:
        expected |= {"reference_features.json", "moving_features.json", "correspondences.json"}
    actual = {path.name for path in directory.iterdir()}
    require(actual == expected, f"Output manifest differs: expected {sorted(expected)}, received {sorted(actual)}")
    for filename in expected:
        if filename.lower().endswith((".jpg", ".png")):
            image = cv2.imread(str(directory / filename), cv2.IMREAD_UNCHANGED)
            require(image is not None and image.size > 0, f"Unreadable or empty output: {filename}")
    return sorted(actual)


def run_native(args, reference_path, moving_path, output, working, *, method="sift",
               estimator="magsac", correspondences=None, dump=False):
    """Run a real entry point with absolute paths from an unrelated working directory."""
    command = [str(args.executable), "--reference", str(reference_path), "--moving", str(moving_path),
               "--output-dir", str(output), "--method", method, "--estimator", estimator,
               "--max-keypoints", "2048", "--resize", "1024", "--device", "cpu",
               "--no-display", "--validate"]
    if correspondences is not None:
        command += ["--correspondences", str(correspondences)]
    if dump:
        command.append("--dump-features")
    if args.model_dir is not None:
        command += ["--model-dir", str(args.model_dir)]
    started = perf_counter()
    process = subprocess.run(command, cwd=working, capture_output=True, text=True, timeout=args.timeout)
    require(process.returncode in (0, 3),
            f"Native CLI failed ({process.returncode}): {process.stdout}\n{process.stderr}")
    report_path = output / "report.json"
    require(report_path.is_file(), "The native CLI did not create report.json")
    report = json.loads(report_path.read_text())
    require(report.get("opencv_version") == "5.1.0", f"Unexpected native OpenCV: {report.get('opencv_version')}")
    accepted = bool(report["accepted"])
    require(process.returncode == (0 if accepted else 3), "CLI exit status disagrees with its acceptance report")
    require(("VALIDATION PASSED" if accepted else "VALIDATION FAILED") in process.stdout,
            "The native CLI omitted its explicit validation marker")
    names = validate_outputs(output, accepted, dump and correspondences is None)
    return report, {"returncode": process.returncode, "output_files": names,
                    "validation_wall_seconds": perf_counter() - started}


def compare_shared_geometry(args, python_result, native_report):
    """Hold every input correspondence fixed when testing geometric implementation parity."""
    require(bool(native_report["accepted"]) == python_result.accepted, "Geometry acceptance differs")
    require(native_report["num_matches"] == python_result.num_matches, "Geometry correspondence counts differ")
    require("inliers" in native_report, "Native report has no inlier mask; rebuild the current C++ source")
    native_mask = np.asarray(native_report["inliers"], dtype=bool)
    require(np.array_equal(native_mask, python_result.inliers), "Estimator inlier masks differ on shared correspondences")
    require(native_report["num_inliers"] == python_result.num_inliers, "Estimator inlier counts differ")
    record = {"accepted": python_result.accepted, "inlier_masks_equal": True,
              "inliers": python_result.num_inliers, "scalar_differences": {}}
    for field, value in [
        ("reference_inlier_hull_coverage", python_result.reference_coverage),
        ("moving_inlier_hull_coverage", python_result.moving_coverage),
        ("projected_reference_visible_fraction", python_result.visible_fraction),
        ("inlier_ratio", python_result.inlier_ratio),
    ]:
        delta = abs(float(native_report[field]) - value)
        record["scalar_differences"][field] = delta
        require(delta <= args.homography_atol, f"{field} differs by {delta}")
    native_H = read_matrix(native_report["H"])
    if python_result.H is None:
        require(native_H is None or native_H.size == 0, "Only the native implementation produced a homography")
        require(native_report["median_reprojection_error_px"] is None, "Absent residual evidence should be null")
        record.update(homography_max_abs_difference=None, quad_max_abs_difference=None,
                      median_reprojection_error_difference=None)
    else:
        require(native_H is not None and native_H.shape == (3, 3), "Native homography is missing or malformed")
        h_delta = maximum_difference(native_H, python_result.H)
        quad_delta = maximum_difference(native_report["projected_reference_quad"], python_result.projected_reference_quad)
        residual_delta = abs(float(native_report["median_reprojection_error_px"])
                             - float(python_result.median_reprojection_error))
        record.update(homography_max_abs_difference=h_delta, quad_max_abs_difference=quad_delta,
                      median_reprojection_error_difference=residual_delta)
        require(h_delta <= args.homography_atol, f"Shared-correspondence H differs by {h_delta}")
        require(quad_delta <= args.geometry_point_atol, f"Projected boundary differs by {quad_delta} pixels")
        require(residual_delta <= args.geometry_point_atol, f"Median inlier residual differs by {residual_delta}")
    return record


def compare_warp_images(output, moving, python_result, reference_shape):
    python_aligned, python_mask = warp_alignment(moving, python_result, reference_shape)
    native_aligned = cv2.imread(str(output / "aligned.png"), cv2.IMREAD_COLOR)
    native_mask = cv2.imread(str(output / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    require(native_aligned.shape == python_aligned.shape and native_mask.shape == python_mask.shape,
            "Warped image or support mask dimensions differ")
    difference = np.abs(native_aligned.astype(np.int16) - python_aligned.astype(np.int16))
    mask_fraction = float(np.mean(native_mask != python_mask))
    record = {"aligned_pixels_equal": bool(np.array_equal(native_aligned, python_aligned)),
              "aligned_max_abs_difference": int(difference.max()),
              "aligned_mean_abs_difference": float(difference.mean()),
              "valid_masks_equal": bool(np.array_equal(native_mask, python_mask)),
              "valid_mask_difference_fraction": mask_fraction}
    # A tiny interpolation difference is reported separately from a wrong transformation.
    require(record["aligned_max_abs_difference"] <= 2 and record["aligned_mean_abs_difference"] <= 0.02,
            f"Warp pixels differ materially: {record}")
    require(mask_fraction <= 1e-4, f"Warp support masks differ materially: {record}")
    return record


def check_geometry(args, working, records):
    rng = np.random.default_rng(411)
    size = (320, 400)
    forward = np.array([[0.94, -0.08, 30], [0.04, 0.96, 8], [0.00012, -0.00008, 1]], np.float64)
    reference_points = rng.uniform([30, 30], [370, 290], (100, 2)).astype(np.float32)
    moving_points = project_points(reference_points, forward)
    moving_points[-30:] = rng.uniform([600, 500], [1000, 900], (30, 2)).astype(np.float32)
    matches = MatchResult(reference_points, moving_points)
    correspondence_path = working / "shared_correspondences.json"
    correspondence_path.write_text(json.dumps({"reference_points": reference_points.tolist(),
                                               "moving_points": moving_points.tolist()}))
    textured = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    textured = cv2.GaussianBlur(textured, (3, 3), 0.6)
    cv2.putText(textured, "GEOMETRY ONLY", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)
    for fixture, reference in [("blank", np.zeros((*size, 3), np.uint8)), ("textured", textured)]:
        moving = cv2.warpPerspective(reference, forward, (size[1], size[0]))
        reference_path, moving_path = working / f"{fixture}_reference.png", working / f"{fixture}_moving.png"
        require(cv2.imwrite(str(reference_path), reference) and cv2.imwrite(str(moving_path), moving),
                "Could not create parity fixture images")
        for estimator in ["ransac", "magsac"]:
            record = {"fixture": fixture, "estimator": estimator, "passed": False}
            records.append(record)
            try:
                output = working / f"geometry-{fixture}-{estimator}"
                native, execution = run_native(args, reference_path, moving_path, output, working,
                                                estimator=estimator, correspondences=correspondence_path)
                expected = estimate_alignment(matches, reference.shape, moving.shape, estimator=estimator)
                record.update(execution=execution, python_geometry=expected.to_dict(),
                              python_inliers=expected.inliers.astype(int).tolist(), native_geometry=native)
                require(expected.accepted, "The shared known-transform fixture was unexpectedly rejected in Python")
                record.update(execution=execution, geometry=compare_shared_geometry(args, expected, native),
                              warp=compare_warp_images(output, moving, expected, reference.shape), passed=True)
                print("GEOMETRY PARITY PASSED", fixture, estimator, flush=True)
            except Exception as error:
                record["error"] = str(error)
                print("GEOMETRY PARITY FAILED", fixture, estimator, error, flush=True)
    # Empty correspondences must remain a real rejection, including exact output cleanup semantics.
    empty_path = working / "empty_correspondences.json"
    empty_path.write_text(json.dumps({"reference_points": [], "moving_points": []}))
    record = {"fixture": "empty_matches", "estimator": "magsac", "passed": False}
    records.append(record)
    try:
        native, execution = run_native(args, reference_path, moving_path, working / "geometry-empty", working,
                                        correspondences=empty_path)
        expected = estimate_alignment(MatchResult(np.empty((0, 2)), np.empty((0, 2))), size, size)
        record.update(execution=execution, python_geometry=expected.to_dict(),
                      python_inliers=expected.inliers.astype(int).tolist(), native_geometry=native)
        record.update(execution=execution, geometry=compare_shared_geometry(args, expected, native), passed=True)
        print("GEOMETRY PARITY PASSED empty_matches", flush=True)
    except Exception as error:
        record["error"] = str(error)
        print("GEOMETRY PARITY FAILED empty_matches", error, flush=True)


def read_features(path):
    data = json.loads(path.read_text())
    points = np.asarray(data["keypoints"], np.float32).reshape(-1, 2)
    descriptors = read_matrix(data["descriptors"])
    if not len(points):
        descriptors = np.empty((0, 128), np.float32)
    require(descriptors.shape == (len(points), 128), f"Malformed native descriptors: {descriptors.shape}")
    require(np.isfinite(points).all() and np.isfinite(descriptors).all(), "Native features contain nonfinite values")
    return data, points, descriptors.astype(np.float32)


def match_feature_points(python_points, native_points, python_descriptors, native_descriptors, point_atol):
    """Associate spatially equal points, resolving repeated SIFT orientations by descriptor."""
    buckets = defaultdict(list)
    for index, point in enumerate(python_points):
        buckets[tuple(np.floor(point / point_atol).astype(np.int64))].append(index)
    used = set()
    associations = []
    for native_index, point in enumerate(native_points):
        cell = np.floor(point / point_atol).astype(np.int64)
        candidates = []
        for offset in product([-1, 0, 1], repeat=2):
            for index in buckets[tuple(cell + offset)]:
                if index not in used and np.linalg.norm(python_points[index] - point) <= point_atol:
                    error = float(np.max(np.abs(python_descriptors[index] - native_descriptors[native_index])))
                    candidates.append((error, index))
        if candidates:
            descriptor_error, index = min(candidates)
            used.add(index)
            associations.append((native_index, index, descriptor_error))
    return associations


def feature_agreement(args, expected, native_path, image):
    data, native_points, native_descriptors = read_features(native_path)
    require((data["image_height"], data["image_width"]) == image.shape[:2],
            "Native feature coordinates refer to different image dimensions")
    require(data["method"] == expected.method, "Feature families differ")
    associations = match_feature_points(expected.keypoints, native_points, expected.descriptors,
                                        native_descriptors, args.point_atol)
    good_descriptors = sum(error <= args.descriptor_atol for _, _, error in associations)
    native_count, python_count = len(native_points), len(expected.keypoints)
    fraction = lambda count, total: count / total if total else float(count == 0)
    point_precision = fraction(len(associations), native_count)
    point_recall = fraction(len(associations), python_count)
    descriptor_precision = fraction(good_descriptors, native_count)
    descriptor_recall = fraction(good_descriptors, python_count)
    errors = [entry[2] for entry in associations]
    point_errors = [float(np.linalg.norm(native_points[a] - expected.keypoints[b])) for a, b, _ in associations]
    same_shape = native_points.shape == expected.keypoints.shape
    record = {"python_keypoints": python_count, "native_keypoints": native_count,
              "spatially_associated_points": len(associations),
              "point_precision": point_precision, "point_recall": point_recall,
              "descriptor_agreement_precision": descriptor_precision,
              "descriptor_agreement_recall": descriptor_recall,
              "associated_point_max_error_px": max(point_errors, default=0.0),
              "associated_descriptor_max_abs_error": max(errors, default=0.0),
              "associated_descriptor_p99_max_abs_error": float(np.percentile(errors, 99)) if errors else 0.0,
              "ordered_keypoint_max_abs_error": maximum_difference(native_points, expected.keypoints) if same_shape else None,
              "ordered_descriptor_max_abs_error": maximum_difference(native_descriptors, expected.descriptors) if same_shape else None}
    passed = min(point_precision, point_recall, descriptor_precision, descriptor_recall) >= args.feature_agreement
    return record, passed


def correspondence_agreement(args, expected, native_path):
    data = json.loads(native_path.read_text())
    reference = np.asarray(data["reference_points"], np.float32).reshape(-1, 2)
    moving = np.asarray(data["moving_points"], np.float32).reshape(-1, 2)
    scores = np.asarray(data["scores"], np.float32).reshape(-1)
    require(reference.shape == moving.shape and len(scores) == len(reference), "Malformed native match arrays")
    expected_pairs = np.column_stack((expected.reference_points, expected.moving_points))
    native_pairs = np.column_stack((reference, moving))
    buckets = defaultdict(list)
    for index, point in enumerate(expected_pairs):
        buckets[tuple(np.floor(point / args.point_atol).astype(np.int64))].append(index)
    used = set()
    associations = []
    for native_index, point in enumerate(native_pairs):
        cell = np.floor(point / args.point_atol).astype(np.int64)
        candidates = []
        for offset in product([-1, 0, 1], repeat=4):
            for index in buckets[tuple(cell + offset)]:
                if index in used:
                    continue
                error = float(np.max(np.abs(point - expected_pairs[index])))
                if error <= args.point_atol:
                    candidates.append((error, index))
        if candidates:
            error, index = min(candidates)
            used.add(index)
            associations.append((native_index, index, error))
    precision = len(associations) / len(native_pairs) if len(native_pairs) else float(not len(expected_pairs))
    recall = len(associations) / len(expected_pairs) if len(expected_pairs) else float(not len(native_pairs))
    score_errors = [abs(float(scores[a]) - float(expected.scores[b])) for a, b, _ in associations]
    record = {"python_matches": len(expected_pairs), "native_matches": len(native_pairs),
              "common_matches": len(associations), "match_precision": precision, "match_recall": recall,
              "common_match_max_coordinate_error_px": max((x[2] for x in associations), default=0.0),
              "common_match_max_score_error": max(score_errors, default=0.0)}
    return record, min(precision, recall) >= args.match_agreement


def check_real_pairs(args, working, assets, records):
    manifest = json.loads((PROJECT / "assets_manifest.json").read_text())
    pairs = {entry["id"]: entry for entry in manifest["pairs"]}
    cases = {
        "box": ("opencv/box.png", "opencv/box_in_scene.png", None),
        "graf_1_6": (pairs["graf_1_6"]["reference"], pairs["graf_1_6"]["moving"],
                     np.asarray(pairs["graf_1_6"]["homography_matrix"], np.float64).reshape(3, 3)),
        "leuven_1_6": (pairs["leuven_1_6"]["reference"], pairs["leuven_1_6"]["moving"],
                       np.asarray(pairs["leuven_1_6"]["homography_matrix"], np.float64).reshape(3, 3)),
        "form": ("learnopencv/form.jpg", "learnopencv/scanned-form.jpg", None),
    }
    for method in args.methods:
        matcher = FeatureMatcher(method, device="cpu", max_keypoints=2048, resize=1024)
        for case_id in args.pairs:
            reference_name, moving_name, H_gt = cases[case_id]
            reference_path, moving_path = assets / reference_name, assets / moving_name
            reference, moving = load_image(reference_path), load_image(moving_path)
            record = {"id": case_id, "method": method, "estimator": "magsac", "passed": False}
            records.append(record)
            try:
                python_reference, python_moving = matcher.extract(reference), matcher.extract(moving)
                matches = matcher.match(python_reference, python_moving)
                expected = estimate_alignment(matches, reference.shape, moving.shape)
                output = working / f"real-{method}-{case_id}"
                native, execution = run_native(args, reference_path, moving_path, output, working,
                                                method=method, dump=True)
                reference_record, reference_pass = feature_agreement(args, python_reference,
                                                                     output / "reference_features.json", reference)
                moving_record, moving_pass = feature_agreement(args, python_moving,
                                                               output / "moving_features.json", moving)
                match_record, match_pass = correspondence_agreement(args, matches, output / "correspondences.json")
                record.update(execution=execution, reference_features=reference_record,
                              moving_features=moving_record, matches=match_record,
                              python_accepted=expected.accepted, native_accepted=bool(native["accepted"]),
                              python_reasons=expected.reasons, native_reasons=native["reasons"])
                if H_gt is not None:
                    native_H = read_matrix(native["H"])
                    record["python_ground_truth_error"] = geometric_error(expected.H, H_gt, reference.shape, moving.shape)
                    record["native_ground_truth_error"] = geometric_error(native_H, H_gt, reference.shape, moving.shape)
                require(reference_pass and moving_pass,
                        f"Feature agreement is below {args.feature_agreement}: {reference_record} / {moving_record}")
                require(match_pass, f"Match-set agreement is below {args.match_agreement}: {match_record}")
                require(expected.accepted == bool(native["accepted"]), "Real-pair acceptance differs")
                if H_gt is not None and expected.H is not None:
                    native_error = record["native_ground_truth_error"]["grid_median_error_px"]
                    python_error = record["python_ground_truth_error"]["grid_median_error_px"]
                    require(native_error is not None and python_error is not None,
                            "A fitted real-pair model has no independent error measurement")
                    record["ground_truth_error_difference_px"] = abs(native_error - python_error)
                    record["python_success_3px"] = bool(expected.accepted and python_error < 3.0)
                    record["native_success_3px"] = bool(native["accepted"] and native_error < 3.0)
                    # Native CPU inference must reproduce the Python CPU behavior,
                    # including an accepted model that independently fails the accuracy criterion.
                    require(record["ground_truth_error_difference_px"] <= args.ground_truth_error_atol,
                            f"Independent error differs: Python {python_error} versus native {native_error}")
                    require(record["python_success_3px"] == record["native_success_3px"],
                            "The two CPU implementations disagree on the independent 3-pixel criterion")
                record["passed"] = True
                print("REAL PIPELINE PARITY PASSED", method, case_id, flush=True)
            except Exception as error:
                record["error"] = str(error)
                print("REAL PIPELINE PARITY FAILED", method, case_id, error, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=["sift", "aliked-nn", "lightglue"], default=["sift"])
    parser.add_argument("--pairs", nargs="+", choices=["box", "graf_1_6", "leuven_1_6", "form"],
                        default=["box", "graf_1_6", "leuven_1_6", "form"])
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--point-atol", type=float, default=0.01)
    parser.add_argument("--descriptor-atol", type=float, default=1e-4)
    parser.add_argument("--geometry-point-atol", type=float, default=1e-5)
    parser.add_argument("--homography-atol", type=float, default=1e-6)
    parser.add_argument("--ground-truth-error-atol", type=float, default=0.05)
    parser.add_argument("--feature-agreement", type=float, default=0.99)
    parser.add_argument("--match-agreement", type=float, default=0.99)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    args.executable = args.executable.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.expanduser().resolve()
    require(args.executable.is_file(), f"Native executable is missing: {args.executable}")
    require(all(value > 0 for value in [args.point_atol, args.descriptor_atol, args.geometry_point_atol,
                                       args.homography_atol, args.ground_truth_error_atol, args.timeout]),
            "Tolerances and timeout must be positive")
    require(0 < args.feature_agreement <= 1 and 0 < args.match_agreement <= 1,
            "Agreement fractions must lie in (0,1]")
    return args


def main():
    args = parse_args()
    require(cv2.__version__ == "5.1.0", f"Authoritative parity requires Python OpenCV 5.1.0, found {cv2.__version__}")
    cv2.setNumThreads(1)
    if any(method != "sift" for method in args.methods) and not args.geometry_only:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(0)
    evidence = {"passed": False, "environment": {"opencv": cv2.__version__, "python": platform.python_version(),
                 "numpy": np.__version__, "platform": platform.platform(), "native_executable": str(args.executable),
                 "native_runtime_uses_python": False}, "thresholds": {
                 "point_atol_px": args.point_atol, "descriptor_atol": args.descriptor_atol,
                 "geometry_point_atol_px": args.geometry_point_atol, "homography_atol": args.homography_atol,
                 "ground_truth_error_atol_px": args.ground_truth_error_atol,
                 "feature_agreement": args.feature_agreement, "match_agreement": args.match_agreement},
                "geometry": [], "real_pairs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="robust-cpp-parity-") as temporary:
            working = Path(temporary)
            check_geometry(args, working, evidence["geometry"])
            if not args.geometry_only:
                assets = download_assets(verbose=False)
                check_real_pairs(args, working, assets, evidence["real_pairs"])
        cases = evidence["geometry"] + evidence["real_pairs"]
        evidence["passed"] = bool(cases) and all(case["passed"] for case in cases)
    except Exception as error:
        evidence["fatal_error"] = str(error)
    finally:
        args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    if not evidence["passed"]:
        print("VALIDATION FAILED: native/Python parity; inspect", args.output, flush=True)
        return 1
    print("VALIDATION PASSED: native/Python parity", len(evidence["geometry"]), "geometry cases and",
          len(evidence["real_pairs"]), "real pipelines;", args.output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

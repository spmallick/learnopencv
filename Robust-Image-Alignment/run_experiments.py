"""Reproduce the article's real-pair benchmark and controlled estimator study.

Ground truth comes from the Oxford capture sequences; the controlled study
creates correspondences with a known transformation and explicitly added noise.
These are small teaching experiments, not a claim of benchmark leadership.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import numpy as np

from download_assets import PROJECT_DIR, download_assets
from robust_image_alignment import (
    AlignmentConfig, FeatureMatcher, MatchResult, estimate_alignment, image_corners,
    load_image, project_points, refine_ecc, warp_alignment,
)


def geometric_error(H, H_reference_to_moving, reference_shape, moving_shape):
    """Compare two mappings on grid points in the annotated overlap.

    Inlier reprojection error evaluates a model on its own selected matches.
    The evaluation grid is independent of the estimated inliers. On captured
    scenes the supplied homography is itself an estimated planar annotation.
    """
    if H is None:
        return {"grid_median_error_px": None, "grid_p95_error_px": None}
    height, width = reference_shape[:2]
    x, y = np.meshgrid(np.linspace(8, width - 9, 30), np.linspace(8, height - 9, 24))
    reference = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    moving = project_points(reference, H_reference_to_moving)
    visible = ((moving[:, 0] >= 8) & (moving[:, 0] < moving_shape[1] - 8)
               & (moving[:, 1] >= 8) & (moving[:, 1] < moving_shape[0] - 8))
    errors = np.linalg.norm(project_points(moving[visible], H) - reference[visible], axis=1)
    corners = image_corners(reference_shape)
    projected_corners = project_points(project_points(corners, H_reference_to_moving), H)
    return {
        "grid_median_error_px": float(np.median(errors)),
        "grid_p95_error_px": float(np.percentile(errors, 95)),
        "grid_points": int(visible.sum()),
        "corner_mean_error_px": float(np.linalg.norm(projected_corners - corners, axis=1).mean()),
    }


def environment(device):
    """Record the actual runtime rather than infer it from installed filenames."""
    import torch
    import torchvision
    import kornia

    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        lines = Path('/proc/cpuinfo').read_text().splitlines()
        cpu = next((s.split(':', 1)[1].strip() for s in lines if s.startswith('model name')), cpu)
    return {
        "opencv": cv2.__version__, "opencv_path": cv2.__file__, "python": sys.version.split()[0],
        "numpy": np.__version__, "torch": torch.__version__,
        "torchvision": torchvision.__version__, "kornia": kornia.__version__,
        "platform": platform.platform(), "cpu": cpu, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else None,
        "opencv_threads": cv2.getNumThreads(), "torch_threads": torch.get_num_threads(),
        "lightglue_commit": json.loads(
            metadata.distribution('lightglue').read_text('direct_url.json') or '{}'
        ).get('vcs_info', {}).get('commit_id'),
        "opencv_release_commit": "e4f78fbcf1a773fbed69e14500e8b734963eb585",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def run_real_pairs(root, device='cuda', repeats=3, max_keypoints=2048):
    """Reuse extracted ALIKED features when comparing the two matching routes."""
    manifest = json.loads((PROJECT_DIR / 'assets_manifest.json').read_text())
    config = AlignmentConfig()
    neural = FeatureMatcher('lightglue', device=device, max_keypoints=max_keypoints)
    nearest = FeatureMatcher('aliked-nn', device=device, max_keypoints=max_keypoints)
    # Share the same ALIKED object as well as the extracted features.
    nearest.extractor = neural.extractor
    sift = FeatureMatcher('sift', max_keypoints=max_keypoints)
    rows = []
    for pair in manifest['pairs']:
        reference, moving = load_image(root / pair['reference']), load_image(root / pair['moving'])
        H_gt = np.asarray(pair['homography_matrix']).reshape(3, 3)
        # One complete warm-up pair per method precedes measured extraction/matching.
        warm_reference, warm_moving = neural.extract(reference), neural.extract(moving)
        warm_neural_matches = neural.match(warm_reference, warm_moving)
        nearest.match(warm_reference, warm_moving)
        sift.match_images(reference, moving)
        for estimator in ['ransac', 'magsac']:
            warm_result = estimate_alignment(warm_neural_matches, reference.shape, moving.shape, config, estimator)
            if warm_result.accepted:
                warp_alignment(moving, warm_result, reference.shape)
        for repetition in range(repeats):
            start = perf_counter()
            reference_neural, moving_neural = neural.extract(reference), neural.extract(moving)
            neural_extraction_wall_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            reference_sift, moving_sift = sift.extract(reference), sift.extract(moving)
            sift_extraction_wall_ms = (perf_counter() - start) * 1000
            matches_by_method = {}
            matching_wall_ms = {}
            for name, model, features in [
                ('sift', sift, (reference_sift, moving_sift)),
                ('aliked-nn', nearest, (reference_neural, moving_neural)),
                ('lightglue', neural, (reference_neural, moving_neural)),
            ]:
                start = perf_counter()
                matches_by_method[name] = model.match(*features)
                matching_wall_ms[name] = (perf_counter() - start) * 1000
            for name, matches in matches_by_method.items():
                for estimator in ['ransac', 'magsac']:
                    start = perf_counter()
                    result = estimate_alignment(matches, reference.shape, moving.shape, config, estimator)
                    geometry_stage_ms = (perf_counter() - start) * 1000
                    warping_ms = 0.0
                    if result.accepted:
                        start = perf_counter()
                        warp_alignment(moving, result, reference.shape)
                        warping_ms = (perf_counter() - start) * 1000
                    error = geometric_error(result.H, H_gt, reference.shape, moving.shape)
                    row = {
                        'id': pair['id'], 'split': pair['split'], 'change': pair['change'],
                        'capture_type': 'real', 'repeat': repetition, 'method': name,
                        'estimator': estimator, **result.to_dict(), **error,
                        'reference_keypoints': matches.reference_keypoints,
                        'moving_keypoints': matches.moving_keypoints,
                        'extraction_ms': neural_extraction_wall_ms if name != 'sift' else sift_extraction_wall_ms,
                        'matching_ms': matching_wall_ms[name], 'geometry_stage_ms': geometry_stage_ms,
                        'warping_ms': warping_ms,
                    }
                    row['pipeline_ms'] = row['extraction_ms'] + row['matching_ms'] + geometry_stage_ms + warping_ms
                    # This is an explicit teaching criterion, not a benchmark's official AUC.
                    row['success_3px'] = bool(result.accepted and error['grid_median_error_px'] is not None
                                                and error['grid_median_error_px'] < 3.0)
                    rows.append(row)
                    if repetition == 0:
                        print(pair['id'], name, estimator, 'accepted=', result.accepted,
                              'grid_error=', error['grid_median_error_px'], flush=True)
        print('Completed', pair['id'], flush=True)
    return rows


def run_estimator_study(seeds=12):
    """Fixed matches isolate estimator configuration from feature matching.

    The synthetic correspondences have spatially distributed support. They do
    not model repeated structures, coherent false matches, or lens distortion.
    """
    H_forward = np.array([[0.87, -0.15, 65.0], [0.08, 0.92, 40.0], [0.00012, -0.00009, 1.]])
    rows = []
    for noise in [0.5, 1.5, 3.0]:
        for seed in range(seeds):
            rng = np.random.default_rng(seed + 1000)
            reference = rng.uniform([50, 50], [750, 550], (160, 2)).astype(np.float32)
            moving = project_points(reference, H_forward)
            moving += rng.normal(0, noise, moving.shape).astype(np.float32)
            # Exactly 60% uniform outliers, shared by both estimators and all thresholds.
            outlier_indices = rng.choice(len(reference), 96, replace=False)
            moving[outlier_indices] = rng.uniform([0, 0], [800, 600], (96, 2))
            matches = MatchResult(reference, moving)
            for threshold in [1.0, 3.0, 5.0]:
                config = AlignmentConfig(reprojection_threshold=threshold,
                                         max_median_reprojection_error=threshold)
                for estimator in ['ransac', 'magsac']:
                    result = estimate_alignment(matches, (600, 800), (600, 800), config, estimator)
                    error = geometric_error(result.H, H_forward, (600, 800), (600, 800))
                    rows.append({'noise_sigma_px': noise, 'outlier_fraction': 0.6, 'seed': seed,
                                 'threshold_px': threshold, 'estimator': estimator,
                                 **result.to_dict(), **error})
    return rows


def run_ecc_study(root, device):
    """Measure refinement against ground truth, including regressions/failures."""
    manifest = json.loads((PROJECT_DIR / 'assets_manifest.json').read_text())
    matcher = FeatureMatcher('lightglue', device=device)
    rows = []
    for pair in manifest['pairs']:
        if pair['id'] not in ['graf_1_3', 'graf_1_6', 'leuven_1_6']:
            continue
        reference, moving = load_image(root / pair['reference']), load_image(root / pair['moving'])
        H_gt = np.asarray(pair['homography_matrix']).reshape(3, 3)
        result = estimate_alignment(matcher.match_images(reference, moving), reference.shape, moving.shape)
        row = {
            'id': pair['id'], 'homography_direction': 'moving_to_reference',
            'initial_H': None if result.H is None else result.H.tolist(),
            'refined_H': None, 'initial_accepted': result.accepted,
            'initial_reasons': result.reasons,
            'before': geometric_error(result.H, H_gt, reference.shape, moving.shape),
            'iterations': 100, 'epsilon': 1e-6,
        }
        if not result.accepted:
            row['status'] = 'initial_estimate_rejected'
        else:
            try:
                start = perf_counter()
                H, correlation = refine_ecc(reference, moving, result.H,
                                            iterations=row['iterations'], epsilon=row['epsilon'])
                row.update(status='returned', correlation=correlation, refined_H=H.tolist(),
                           refinement_ms=(perf_counter() - start)*1000,
                           after=geometric_error(H, H_gt, reference.shape, moving.shape))
            except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                row.update(status='failed', error=str(error))
        rows.append(row)
    return rows


def check_backend(root, device):
    """Evaluate each held-out pair once and change only point pruning on Graf6."""
    import torch

    manifest = json.loads((PROJECT_DIR / 'assets_manifest.json').read_text())
    matcher = FeatureMatcher('lightglue', device=device)
    network = matcher.matcher
    default_width = network.conf.width_confidence
    rows = []
    for pair in manifest['pairs']:
        if pair['split'] != 'evaluation':
            continue
        reference = load_image(root / pair['reference'])
        moving = load_image(root / pair['moving'])
        # Cache these exact features even when running the matched-pair ablation.
        reference_features = matcher.extract(reference)
        moving_features = matcher.extract(moving)
        widths = [default_width, -1] if pair['id'] == 'graf_1_6' else [default_width]
        for width in widths:
            network.conf.width_confidence = width
            matches = matcher.match(reference_features, moving_features)
            result = estimate_alignment(matches, reference.shape, moving.shape)
            H_gt = np.asarray(pair['homography_matrix']).reshape(3, 3)
            error = geometric_error(result.H, H_gt, reference.shape, moving.shape)
            rows.append({
                'id': pair['id'], 'width_confidence': width,
                'reference_keypoints': matches.reference_keypoints,
                'moving_keypoints': matches.moving_keypoints,
                **result.to_dict(), **error,
                'success_3px': bool(result.accepted and error['grid_median_error_px'] < 3.0),
            })
        network.conf.width_confidence = default_width
    return {
        'environment': environment(device),
        'method': 'aliked-n16 + LightGlue + USAC_MAGSAC',
        'configuration': asdict(AlignmentConfig()),
        'default_width_confidence': default_width,
        'depth_confidence': network.conf.depth_confidence,
        'filter_threshold': network.conf.filter_threshold,
        'prune_if_keypoint_count_above': network.pruning_min_kpts(torch.device(device)),
        'max_keypoints': matcher.max_keypoints, 'resize': matcher.resize,
        'results': rows,
    }


def write_real_pair_tables(data, directory):
    """Export the same raw rows and common successful-pair timings used in the article."""
    rows = data['real_pairs']
    fields = ('id', 'split', 'repeat', 'method', 'estimator', 'accepted', 'success_3px',
              'grid_median_error_px', 'grid_p95_error_px', 'num_matches', 'num_inliers',
              'extraction_ms', 'matching_ms', 'geometry_stage_ms', 'warping_ms', 'pipeline_ms')
    with (directory / 'real_pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    evaluation = [row for row in rows if row['split'] == 'evaluation' and row['estimator'] == 'magsac']
    pairs = sorted({row['id'] for row in evaluation})
    methods = ('sift', 'aliked-nn', 'lightglue')
    # Repeats measure timing. A passing pair must pass in every measured trial.
    successes = {
        method: {pair for pair in pairs if sum(row['success_3px'] for row in evaluation
                                               if row['id'] == pair and row['method'] == method)
                 == data['protocol']['repeats']}
        for method in methods
    }
    common = set.intersection(*successes.values())
    pipelines = []
    for method in methods:
        times = [row['pipeline_ms'] for row in evaluation if row['id'] in common and row['method'] == method]
        pipelines.append({'method': method, 'successes': len(successes[method]), 'pairs': len(pairs),
                          'timing_trials': len(times),
                          'pipeline_ms_p10_p50_p90': np.percentile(times, [10, 50, 90]).tolist() if times else None})
    summary = {'opencv': data['environment']['opencv'], 'evaluation_pairs': len(pairs),
               'evaluation_scenes': len({row['id'].split('_')[0] for row in evaluation}),
               'validation_pairs': len({row['id'] for row in rows if row['split'] == 'validation'}),
               'repeats_per_pair': data['protocol']['repeats'], 'timing_common_success_pairs': sorted(common),
               'pipelines': pipelines}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--output', type=Path, help='Destination JSON file.')
    parser.add_argument('--check-backend', action='store_true',
                        help='Run the eight held-out LightGlue pairs and the Graf6 pruning comparison.')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--study-seeds', type=int, default=12)
    parser.add_argument('--allow-other-opencv', action='store_true', help='Smoke tests only; not publication evidence.')
    args = parser.parse_args()
    if args.repeats < 1 or args.study_seeds < 1:
        parser.error('Repeats and study seeds must be positive.')
    if args.output is None:
        name = f'backend_{args.device}.json' if args.check_backend else 'experiments.json'
        args.output = PROJECT_DIR / 'results' / name
    if not args.allow_other_opencv and cv2.__version__ != '5.1.0':
        raise RuntimeError(f'Publication experiments require OpenCV 5.1.0, found {cv2.__version__}')
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    root = download_assets(verbose=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.check_backend:
        args.output.write_text(json.dumps(check_backend(root, args.device), indent=2, allow_nan=False) + '\n')
        print('Saved', args.output, flush=True)
        return
    data = {'environment': environment(args.device),
            'protocol': {'max_keypoints': 2048, 'resize_longest_side_cap': 1024,
                         'repeats': args.repeats, 'config': asdict(AlignmentConfig()),
                         'timing_excludes': ['disk_io', 'model_load', 'warm_up', 'visualization'],
                         'pipeline_timing': 'both image extractions + matching + estimation/validation + image/mask warping when accepted',
                         'success': 'accepted and median ground-truth overlap-grid error < 3 original reference pixels'},
            'real_pairs': run_real_pairs(root, args.device, args.repeats)}
    # Save each completed study so a later optional refinement cannot erase prior evidence.
    args.output.write_text(json.dumps(data, indent=2))
    data['estimator_study'] = run_estimator_study(args.study_seeds)
    args.output.write_text(json.dumps(data, indent=2))
    data['ecc_study'] = run_ecc_study(root, args.device)
    args.output.write_text(json.dumps(data, indent=2))
    write_real_pair_tables(data, args.output.parent)
    args.output.with_name('opencv_build.txt').write_text(cv2.getBuildInformation().strip() + '\n')
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()

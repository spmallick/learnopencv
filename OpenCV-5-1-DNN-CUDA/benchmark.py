"""Measure one ImageNet classifier/backend in a fresh process and verify its placement."""

import argparse
from collections import Counter
import hashlib
import json
from math import lcm
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
from time import perf_counter_ns


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True, help='single-input, single-output ONNX file')
    p.add_argument('--image', type=Path, action='append', required=True, help='repeat for multiple images')
    p.add_argument('--labels', type=Path, help='class names, one per line in output order')
    p.add_argument('--backend', choices=('opencv-cpu', 'opencv-cuda', 'ort-cuda'), required=True)
    p.add_argument('--output', type=Path, required=True, help='new or empty result directory')
    p.add_argument('--batches', default='1', help='batch sizes in request order, e.g. 1,2')
    p.add_argument('--tf32', choices=('on', 'off'), default='off', help='allow TF32 on CUDA')
    p.add_argument('--threads', type=int, default=6, help='OpenCV threads and ORT intra-op threads')
    p.add_argument('--resize-short', type=int, default=256, help='short edge before center crop to 224')
    p.add_argument('--warmup', type=int, default=10, help='discard this many forwards before sampling')
    p.add_argument('--runs', type=int, default=50, help='requests in each measured loop')
    p.add_argument('--worker', choices=('time', 'proof'), help=argparse.SUPPRESS)
    args = p.parse_args()
    try:
        args.batches = [int(s) for s in args.batches.split(',')]
    except ValueError:
        p.error('--batches must be comma-separated positive integers')
    if min(args.batches) < 1 or min(args.runs, args.warmup, args.threads) < 1:
        p.error('batch sizes, runs, warmup, and threads must all be positive')
    if args.backend == 'opencv-cpu' and args.tf32 == 'on':
        p.error('TF32 is a CUDA setting')
    if args.resize_short < 224:
        p.error('--resize-short must be at least 224')
    for path in [args.model, *args.image] + ([args.labels] if args.labels else []):
        if not path.is_file():
            p.error(f'File not found: {path}')
    return args


def case_paths(args):
    # Cycle through sizes and image order independently; two B=1 requests can
    # carry different pixels. Repeating the list covers one full request cycle.
    cases = []
    for i in range(lcm(len(args.image), len(args.batches))):
        batch = args.batches[i % len(args.batches)]
        cases.append([args.image[(i + j) % len(args.image)] for j in range(batch)])
    return cases


def gpu_memory():
    # Query the worker's own retained GPU memory with the live session still open.
    # Other users of the GPU do not count. A container may need --pid=host.
    cmd = ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
           '--format=csv,noheader,nounits']
    try:
        rows = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    for line in rows.splitlines():
        fields = [value.strip() for value in line.split(',')]
        if len(fields) == 3 and fields[0] == str(os.getpid()) and fields[2].isdigit():
            return int(fields[2]), fields[1]
    return None, None


def summary(samples, batches):
    import numpy as np
    return {'p50_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95)),
            'samples_ms': samples, 'batches': batches}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def worker(args):
    # Imports and fixture preparation deliberately precede the cold-session timer.
    import cv2 as cv
    import numpy as np
    from classifier import check_scores, make_blob, opencv_runner, ort_runner, top_five

    if cv.__version__.split('-')[0] != '5.1.0':
        raise RuntimeError(f'This benchmark targets OpenCV 5.1.0; imported {cv.__version__}')
    cv.setNumThreads(args.threads)
    ort = None
    if args.backend == 'ort-cuda':
        import onnxruntime as ort

    labels = args.labels.read_text(encoding='utf-8').splitlines() if args.labels else []
    cases = case_paths(args)
    blobs = [make_blob(paths, args.resize_short) for paths in cases]
    t0 = perf_counter_ns()
    if ort:
        forward, session = ort_runner(args, ort, profile=args.worker == 'proof')
    else:
        forward = opencv_runner(args)
    t1 = perf_counter_ns()
    scores = forward(blobs[0])
    t2 = perf_counter_ns()
    check_scores(scores, len(cases[0]), labels)

    if args.worker == 'proof':
        # Exercise every shape/input combination in an untimed sibling process.
        for blob in blobs:
            forward(blob)
        if ort:
            Path(session.end_profiling()).replace(args.output / 'placement-profile.json')
        return

    cold = {'load_ms': (t1 - t0) / 1e6, 'first_forward_ms': (t2 - t1) / 1e6,
            'load_to_first_result_ms': (t2 - t0) / 1e6}
    for i in range(args.warmup):
        forward(blobs[i % len(cases)])

    # Time each request, including host/device copies and host-visible output.
    # Continue the same cycle across both loops, even for odd warm-up counts.
    phase = args.warmup
    stats = {}
    for metric in ['forward', 'end_to_end']:
        samples, batches = [], []
        for i in range(args.runs):
            k = (phase + i) % len(cases)
            start = perf_counter_ns()
            if metric == 'forward':
                scores = forward(blobs[k])
            else:
                image_batch = make_blob(cases[k], args.resize_short)
                scores = forward(image_batch)
                top_five(scores, labels)
            samples.append((perf_counter_ns() - start) / 1e6)
            batches.append(len(cases[k]))
        stats[metric] = summary(samples, batches)
        phase += args.runs

    # Save full logits, outside the timer, for cross-model-export and runtime checks.
    predictions, arrays = [], {}
    for k, paths in enumerate(cases):
        key = f'case_{k}'
        scores = forward(blobs[k])
        check_scores(scores, len(paths), labels)
        arrays[key] = scores.copy()
        ids, names = top_five(scores, labels)
        predictions.append({'key': key, 'images': [path.name for path in paths],
                            'batch': len(paths), 'top5_ids': ids, 'top1_labels': names})
    gpu_mib, uuid = gpu_memory() if args.backend.endswith('cuda') else (None, None)
    # Linux returns KiB, macOS bytes. This high-water mark covers the worker process.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1048576 if sys.platform == 'darwin' else 1024)
    np.savez(args.output / 'logits.npz', **arrays)
    data = {'backend': args.backend, 'model': args.model.name, 'tf32': args.tf32,
            'batches': args.batches, 'threads': args.threads,
            'runs': args.runs, 'warmup': args.warmup, 'cold': cold, **stats,
            'opencv': cv.__version__, 'onnxruntime': ort.__version__ if ort else None,
            'memory': {'peak_rss_mib': rss_mib, 'gpu_after_warm_mib': gpu_mib, 'gpu_uuid': uuid},
            'predictions': predictions}
    write_json(args.output / 'timing.json', data)


def placement_report(backend, log, profile_path):
    if backend == 'opencv-cuda':
        # This INFO text is from Net::Impl::finalizeGraph in the pinned 5.1.0 release.
        rows = re.findall(r"finalize op #(\d+) '(.*?)' \(([^)]+)\) -> (CPU|CUDA)", log)
        # A shape change may finalize the same graph again. Count each named node
        # and placement once, just as the ORT branch deduplicates profile events.
        ops = [(kind, device) for _, _, kind, device in sorted(set(rows))]
        blockers = [line for line in log.splitlines() if 'running the whole graph on CPU' in line]
        if blockers or not any(device == 'CUDA' for _, device in ops):
            detail = '\n'.join(blockers) or 'No finalized CUDA operators in placement.log'
            raise RuntimeError(f'OpenCV CUDA placement check failed: {detail}')
    else:
        events = json.loads(profile_path.read_text(encoding='utf-8'))
        # Count unique executed nodes across repeated forwards in ORT's profiling run.
        nodes = {(e['name'], e['args']['op_name'], e['args']['provider']) for e in events
                 if e.get('cat') == 'Node' and e.get('args', {}).get('provider')}
        ops = [(kind, 'CUDA' if provider == 'CUDAExecutionProvider' else provider)
               for _, kind, provider in sorted(nodes)]
        if not any(device == 'CUDA' and not kind.startswith('Memcpy') for kind, device in ops):
            raise RuntimeError('ORT profile did not show a compute node executed by CUDA')
    cpu = Counter(kind for kind, device in ops if device in ('CPU', 'CPUExecutionProvider'))
    return {'cuda_ops': sum(device == 'CUDA' for _, device in ops),
            'cpu_ops': sum(cpu.values()), 'cpu_op_types': dict(cpu), 'whole_graph_cpu': False}


def run(args):
    # Reject old output so that a failed retry can never expose stale successful results.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f'Choose a new or empty --output directory: {args.output}')
    phases = ['time', 'proof'] if args.backend.endswith('cuda') else ['time']
    for phase in phases:
        log = args.output / ('placement.log' if phase == 'proof' else 'timing.log')
        env = dict(os.environ, OPENCV_LOG_LEVEL='INFO' if phase == 'proof' else 'WARNING')
        cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker', phase]
        with log.open('w') as stream:
            completed = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            tail = '\n'.join(log.read_text(errors='replace').splitlines()[-10:])
            raise RuntimeError(f'{phase} worker failed ({completed.returncode}); see {log}\n{tail}')

    data = json.loads((args.output / 'timing.json').read_text())
    if 'proof' in phases:
        data['placement'] = placement_report(args.backend, log.read_text(errors='replace'),
                                             args.output / 'placement-profile.json')
    else:
        data['placement'] = {'requested': 'CPU'}
    with args.model.open('rb') as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    data['model_sha256'] = digest.hexdigest()
    data['resize_short'] = args.resize_short
    data['cuda_visible_devices'] = os.getenv('CUDA_VISIBLE_DEVICES')
    data['nvidia_tf32_override'] = os.getenv('NVIDIA_TF32_OVERRIDE')
    # Write the public result only after every placement check succeeds.
    tmp = args.output / 'result.tmp'
    write_json(tmp, data)
    tmp.replace(args.output / 'result.json')
    (args.output / 'timing.json').unlink()
    print(f"{args.backend}: {data['forward']['p50_ms']:.2f} ms forward, "
          f"{data['end_to_end']['p50_ms']:.2f} ms end to end (p50)")
    for p in data['predictions']:
        predictions = p['top1_labels'] or [str(row[0]) for row in p['top5_ids']]
        print(', '.join(f'{image} → {label}' for image, label in zip(p['images'], predictions)))
    print(args.output / 'result.json')


if __name__ == '__main__':
    parsed = parse_args()
    try:
        if parsed.worker:
            worker(parsed)
        else:
            run(parsed)
    except (ValueError, RuntimeError) as err:
        sys.exit(str(err))

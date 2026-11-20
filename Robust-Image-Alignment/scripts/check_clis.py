"""Exercise all matching/estimator CLI combinations from an unrelated directory."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile

import cv2


PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / 'assets' / 'downloads'
records = []
assert cv2.__version__ == '5.1.0', cv2.__version__
with tempfile.TemporaryDirectory(prefix='alignment-cli-matrix-') as unrelated:
    for method in ['sift', 'aliked-nn', 'lightglue']:
        for estimator in ['ransac', 'magsac']:
            output = Path(unrelated) / f'{method}-{estimator}'
            command = [sys.executable, str(PROJECT / 'robust_image_alignment.py'),
                       '--reference', str(ASSETS / 'opencv/box.png'),
                       '--moving', str(ASSETS / 'opencv/box_in_scene.png'),
                       '--output-dir', str(output), '--method', method,
                       '--estimator', estimator, '--device', 'cpu', '--no-display', '--validate']
            process = subprocess.run(command, cwd=unrelated, capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stdout + process.stderr)
            assert 'VALIDATION PASSED' in process.stdout
            names = sorted(path.name for path in output.iterdir())
            assert names == sorted(['report.json', 'matches.jpg', 'aligned.png',
                                    'valid_mask.png', 'overlay.jpg', 'reference_boundary.jpg']), names
            report = json.loads((output / 'report.json').read_text())
            assert report['opencv_version'] == '5.1.0' and report['accepted']
            for name in names:
                if name != 'report.json':
                    image = cv2.imread(str(output / name), cv2.IMREAD_UNCHANGED)
                    assert image is not None and image.size
            records.append({'method': method, 'estimator': estimator, 'device': report['device'],
                            'opencv': report['opencv_version'], 'accepted': True,
                            'inliers': report['num_inliers'], 'output_files': names})
            print('CLI VALIDATION PASSED', method, estimator, flush=True)
target = PROJECT / 'results' / 'cli-validation.json'
target.write_text(json.dumps(records, indent=2))
print('Saved', target)

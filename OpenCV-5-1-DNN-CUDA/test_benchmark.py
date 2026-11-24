"""Small regression checks for preprocessing and the CUDA placement gate."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2 as cv
import numpy as np

from benchmark import placement_report
from classifier import make_blob


class BenchmarkTests(unittest.TestCase):
    def test_rgb_normalization_and_batch_order(self):
        # Distinct channels detect both an accidental BGR blob and reversed means.
        colors = np.array([[3, 63, 130], [202, 19, 22]], np.uint8)
        with TemporaryDirectory() as root:
            paths = []
            for i, color in enumerate(colors):
                path = Path(root) / f'{i}.png'
                self.assertTrue(cv.imwrite(str(path), np.full((224, 320, 3), color, np.uint8)))
                paths.append(path)
            blob = make_blob(paths)
        self.assertEqual(blob.shape, (2, 3, 224, 224))
        expected_rgb = colors[:, ::-1] / 255
        expected = (expected_rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np.testing.assert_allclose(blob[:, :, 100, 190], expected, atol=5e-7)

    def test_opencv_rejects_fallback_even_when_a_cuda_line_is_present(self):
        with self.assertRaisesRegex(RuntimeError, 'whole graph on CPU'):
            placement_report('opencv-cuda', "finalize op #1 'conv' (Conv2) -> CUDA\n"
                             'running the whole graph on CPU', None)
        with self.assertRaisesRegex(RuntimeError, 'No finalized CUDA'):
            placement_report('opencv-cuda', "finalize op #1 'conv' (Conv2) -> CPU", None)

    def test_repeated_finalization_preserves_one_placement_per_node(self):
        lines = "finalize op #1 'qkv' (MatMul) -> CUDA\nfinalize op #2 's' (Shape) -> CPU\n"
        report = placement_report('opencv-cuda', lines * 2, None)
        self.assertEqual(report['cuda_ops'], 1)
        self.assertEqual(report['cpu_op_types'], {'Shape': 1})

    def test_ort_requires_compute_on_cuda(self):
        events = [{'cat': 'Node', 'name': 'copy', 'args': {
            'op_name': 'MemcpyFromHost', 'provider': 'CUDAExecutionProvider'}}]
        with TemporaryDirectory() as root:
            path = Path(root) / 'profile.json'
            path.write_text(json.dumps(events))
            with self.assertRaisesRegex(RuntimeError, 'compute node'):
                placement_report('ort-cuda', '', path)
            events.append({'cat': 'Node', 'name': 'matmul', 'args': {
                'op_name': 'MatMul', 'provider': 'CUDAExecutionProvider'}})
            path.write_text(json.dumps(events * 2))
            self.assertEqual(placement_report('ort-cuda', '', path)['cuda_ops'], 2)


if __name__ == '__main__':
    unittest.main()

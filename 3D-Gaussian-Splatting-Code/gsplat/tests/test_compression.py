"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import pytest
import torch

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_png_compression():
    from gsplat.compression import PngCompression

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100000
    splats = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.randn(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.randn(N),
            "sh0": torch.randn(N, 1, 3),
            "shN": torch.randn(N, 24, 3),
            "features": torch.randn(N, 128),
        }
    ).to(device)
    compress_dir = "/tmp/gsplat/compression"

    compression_method = PngCompression()
    # run compression and save the compressed files to compress_dir
    compression_method.compress(compress_dir, splats)
    # decompress the compressed files
    splats_c = compression_method.decompress(compress_dir)


if __name__ == "__main__":
    test_png_compression()

"""Fetch the exact pretrained weights used by the article into Torch's cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request


MODELS = [
    {
        'filename': 'aliked-n16.pth',
        'url': 'https://raw.githubusercontent.com/Shiaoming/ALIKED/683d7c65197395c0b3f01ebe76e1084a27e73a65/models/aliked-n16.pth',
        'sha256': '5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2',
        'bytes': 2738091,
    },
    {
        'filename': 'aliked_lightglue_v0-1_arxiv.pth',
        'url': 'https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth',
        'sha256': 'd975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503',
        'bytes': 47632827,
    },
]


def download_models(verbose=True):
    """Verify cached files or atomically replace them with the pinned payloads."""
    import torch
    cache = Path(torch.hub.get_dir()) / 'checkpoints'
    cache.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        path = cache / model['filename']
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == model['sha256']:
            continue
        if verbose:
            print('Downloading verified weights:', model['filename'])
        with urllib.request.urlopen(model['url'], timeout=60) as response:
            payload = response.read()
        if len(payload) != model['bytes'] or hashlib.sha256(payload).hexdigest() != model['sha256']:
            raise RuntimeError('Model payload does not match its recorded SHA-256: ' + model['filename'])
        temporary = path.with_suffix('.download')
        temporary.write_bytes(payload)
        temporary.replace(path)
    if verbose:
        print('Verified both pretrained checkpoints in', cache)
    return cache


if __name__ == '__main__':
    download_models()

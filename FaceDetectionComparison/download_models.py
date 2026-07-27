#!/usr/bin/env python3
"""Download the pinned dynamic YuNet ONNX model with SHA-256 verification."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


ZOO_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_NAME = "face_detection_yunet_2026may.onnx"
MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
MODEL_SIZE = 229_738
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{ZOO_COMMIT}/models/face_detection_yunet/{MODEL_NAME}"
)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "models" / MODEL_NAME


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    """Match the model against the pinned Git LFS size and object digest."""

    actual_size = path.stat().st_size
    if actual_size != MODEL_SIZE:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {MODEL_SIZE}, got {actual_size}."
        )
    actual_digest = sha256_file(path)
    if actual_digest != MODEL_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: "
            f"expected {MODEL_SHA256}, got {actual_digest}."
        )


def download(output: Path, force: bool = False) -> Path:
    """Download to a temporary neighbor and atomically install verified bytes."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        verify(output)
        print(f"Verified existing model: {output}")
        return output

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MODEL_NAME}.", suffix=".download", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        print(f"Downloading {MODEL_NAME} from pinned OpenCV Zoo commit {ZOO_COMMIT}")
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        verify(temporary_path)
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Verified SHA-256 {MODEL_SHA256}")
    print(f"Saved model: {output}")
    return output


def parse_args() -> argparse.Namespace:
    """Parse destination and intentional refresh controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download(arguments.output, arguments.force)

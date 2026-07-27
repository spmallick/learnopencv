#!/usr/bin/env python3
"""Download the pinned OpenCV Zoo MediaPipe Pose model with SHA-256 checking."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


ZOO_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_NAME = "pose_estimation_mediapipe_2023mar.onnx"
MODEL_SHA256 = "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
MODEL_SIZE = 5_557_238
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    f"{ZOO_COMMIT}/models/pose_estimation_mediapipe/{MODEL_NAME}"
)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "models" / MODEL_NAME


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks so large model files do not fill memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    """Reject truncated, substituted, or otherwise unexpected model bytes."""

    size = path.stat().st_size
    if size != MODEL_SIZE:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {MODEL_SIZE}, got {size}."
        )
    actual = sha256_file(path)
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {MODEL_SHA256}, got {actual}."
        )


def download(output: Path, force: bool = False) -> Path:
    """Download atomically, then verify before making the model available."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        verify(output)
        print(f"Verified existing model: {output}")
        return output

    # Keep an incomplete network transfer outside the final path. os.replace is
    # atomic on the same filesystem, so readers never observe a partial model.
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
    """Parse a destination override useful for CI and offline caches."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when a valid model already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download(arguments.output, arguments.force)

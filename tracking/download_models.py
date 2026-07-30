#!/usr/bin/env python3
"""Download and verify the ONNX files used by OpenCV's DNN trackers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request


MODELS_DIR = Path(__file__).resolve().parent / "models"
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# Every URL is pinned to an upstream revision. Exact byte counts and SHA-256
# digests are checked before a temporary download can replace a model file.
MODEL_GROUPS = {
    "dasiamrpn": {
        "dasiamrpn_model.onnx": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "fef72f8fa7c52eaf116d3df358d24e6e959ada0e/"
            "models/object_tracking_dasiamrpn/"
            "object_tracking_dasiamrpn_model_2021nov.onnx",
            "e88370b85cbad914a5eb414d9d9e0820f87fd0cd89b65205a766174206c35719",
            91040894,
        ),
        "dasiamrpn_kernel_cls1.onnx": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "fef72f8fa7c52eaf116d3df358d24e6e959ada0e/"
            "models/object_tracking_dasiamrpn/"
            "object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx",
            "d85b03e2aeded6cc9be945dfdc3ed6b8f4151f101e485037b6c5d5b36a6c4204",
            23603598,
        ),
        "dasiamrpn_kernel_r1.onnx": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "fef72f8fa7c52eaf116d3df358d24e6e959ada0e/"
            "models/object_tracking_dasiamrpn/"
            "object_tracking_dasiamrpn_kernel_r1_2021nov.onnx",
            "082c85d231b88b97a1b2a50e73b640a332c5d98d7c1d80b5da9ab534fa7a9e5b",
            47206788,
        ),
    },
    "nano": {
        "nanotrack_backbone_sim.onnx": (
            "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/"
            "248663fde6bf7c40190cf10ee396d5662919ecd3/"
            "NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx",
            "530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4",
            1056849,
        ),
        "nanotrack_head_sim.onnx": (
            "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/"
            "248663fde6bf7c40190cf10ee396d5662919ecd3/"
            "NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx",
            "0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9",
            726198,
        ),
    },
    "vit": {
        "object_tracking_vittrack_2023sep.onnx": (
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
            "47534e27c9851bb1128ccc0102f1145e27f23f98/"
            "models/object_tracking_vittrack/"
            "object_tracking_vittrack_2023sep.onnx",
            "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d",
            714726,
        ),
    },
}


def sha256_of(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(
    name: str,
    url: str,
    expected_sha: str,
    expected_size: int,
    *,
    force: bool,
    models_dir: Path,
) -> bool:
    """Atomically download and verify one model."""
    destination = models_dir / name
    if destination.exists() and not force:
        try:
            existing_size = destination.stat().st_size
            existing_sha = (
                sha256_of(destination) if existing_size == expected_size else None
            )
        except OSError as error:
            print(f"[warn] cannot verify existing {name}: {error}")
        else:
            if existing_size == expected_size and existing_sha == expected_sha:
                print(f"[skip] {name} already verified")
                return True
            print(f"[warn] {name} is invalid; downloading a verified replacement")

    temporary_path: Path | None = None
    print(f"[get ] {name}")
    try:
        with tempfile.NamedTemporaryFile(
            dir=models_dir,
            prefix=f".{name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        request = urllib.request.Request(
            url, headers={"User-Agent": "LearnOpenCV-model-downloader/1.0"}
        )
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised_size = int(content_length)
                except ValueError:
                    advertised_size = None
                if advertised_size is not None and advertised_size > expected_size:
                    print(
                        f"[fail] {name}: server advertised {advertised_size} bytes; "
                        f"expected {expected_size}"
                    )
                    return False

            downloaded_size = 0
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    downloaded_size += len(chunk)
                    if downloaded_size > expected_size:
                        print(
                            f"[fail] {name}: response exceeded "
                            f"{expected_size} bytes"
                        )
                        return False
                    output.write(chunk)

        if downloaded_size != expected_size:
            print(
                f"[fail] {name}: received {downloaded_size} bytes; "
                f"expected {expected_size}"
            )
            return False
        actual_sha = sha256_of(temporary_path)
        if actual_sha != expected_sha:
            print(f"[fail] {name}: checksum mismatch ({actual_sha})")
            return False
        temporary_path.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        print(f"[fail] {name}: {error}")
        return False
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"[warn] cannot remove partial file {temporary_path}: {error}")

    print(f"[ ok ] {name} sha256={expected_sha}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracker",
        choices=("all", *MODEL_GROUPS),
        default="all",
        help="download one tracker model set or all three (default: all)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"destination directory (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download again even when a verified file is already present",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        arguments.models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"[fail] cannot create {arguments.models_dir}: {error}")
        return 1

    groups = MODEL_GROUPS if arguments.tracker == "all" else {
        arguments.tracker: MODEL_GROUPS[arguments.tracker]
    }
    results = []
    for files in groups.values():
        for name, (url, expected_sha, expected_size) in files.items():
            results.append(
                download_one(
                    name,
                    url,
                    expected_sha,
                    expected_size,
                    force=arguments.force,
                    models_dir=arguments.models_dir,
                )
            )
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

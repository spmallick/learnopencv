#!/usr/bin/env python3
"""Download the ONNX models required by the DNN-based OpenCV trackers.

The classical trackers (MIL, KCF, CSRT) need no model files, so this script
only fetches the assets for TrackerDaSiamRPN, TrackerNano, and TrackerVit.
Every file lands in ``<repo>/Object-Tracking-OpenCV5/models/`` where both the
Python and C++ examples expect to find it.

Usage:
    python3 download_models.py            # download everything that is missing
    python3 download_models.py --force    # re-download even if present
    python3 download_models.py --models-dir /tmp/tracker-models
"""

# argparse builds the small command-line interface for this script.
import argparse
# hashlib computes SHA-256 checksums so a truncated or tampered download fails loudly.
import hashlib
# pathlib gives us robust, OS-independent path handling anchored at this file.
from pathlib import Path
# sys provides the exit code used to signal success or failure to callers.
import sys
# tempfile creates same-directory partial files for atomic replacement.
import tempfile
# urllib.error supplies the download-specific exception classes.
import urllib.error
# urllib.request performs the actual HTTP downloads using only the standard library.
import urllib.request

# Resolve the models directory relative to this script, never the caller's
# current working directory, so the script works from anywhere.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Bound each network read and stream in modest chunks so a stalled or malicious
# response cannot hang indefinitely or consume model-sized memory.
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# Each entry maps the local filename to an immutable upstream URL and its
# expected SHA-256 and byte size. A download never replaces a model until both
# pinned properties match.
MODELS = {
    # --- TrackerNano (NanoTrack v2): a ~2 MB two-file siamese tracker. ---
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
    # --- TrackerVit: the transformer tracker from the OpenCV model zoo. ---
    # The zoo stores the file in Git LFS, so we fetch through the media
    # endpoint at an immutable commit, which serves the real payload instead
    # of the LFS pointer.
    "object_tracking_vittrack_2023sep.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "47534e27c9851bb1128ccc0102f1145e27f23f98/"
        "models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
        "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d",
        714726,
    ),
    # --- TrackerDaSiamRPN: three files from the immutable OpenCV Zoo revision
    # documented by OpenCV 5's samples/dnn/models.yml. The local names retain
    # the defaults expected by cv::TrackerDaSiamRPN and the Python binding. ---
    "dasiamrpn_model.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "fef72f8fa7c52eaf116d3df358d24e6e959ada0e/"
        "models/object_tracking_dasiamrpn/"
        "object_tracking_dasiamrpn_model_2021nov.onnx",
        "e88370b85cbad914a5eb414d9d9e0820f87fd0cd89b65205a766174206c35719",
        91040894,
    ),
    "dasiamrpn_kernel_r1.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "fef72f8fa7c52eaf116d3df358d24e6e959ada0e/"
        "models/object_tracking_dasiamrpn/"
        "object_tracking_dasiamrpn_kernel_r1_2021nov.onnx",
        "082c85d231b88b97a1b2a50e73b640a332c5d98d7c1d80b5da9ab534fa7a9e5b",
        47206788,
    ),
    "dasiamrpn_kernel_cls1.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
        "fef72f8fa7c52eaf116d3df358d24e6e959ada0e/"
        "models/object_tracking_dasiamrpn/"
        "object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx",
        "d85b03e2aeded6cc9be945dfdc3ed6b8f4151f101e485037b6c5d5b36a6c4204",
        23603598,
    ),
}


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, reading in 1 MiB chunks."""
    # Stream the file instead of loading it fully so large models stay cheap.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(
        name: str, url: str, expected_sha: str, expected_size: int,
        force: bool, models_dir: Path) -> bool:
    """Atomically download and verify one model. Return True on success."""
    destination = models_dir / name
    # Skip files that already exist and pass verification unless forced.
    if destination.exists() and not force:
        try:
            existing_size = destination.stat().st_size
            existing_sha = (
                sha256_of(destination)
                if existing_size == expected_size else None
            )
        except OSError as error:
            print(f"[warn] cannot verify existing {name}: {error}")
        else:
            if existing_size == expected_size and existing_sha == expected_sha:
                print(f"[skip] {name} already present")
                return True
            print(f"[warn] {name} exists but fails checksum; re-downloading")
    print(f"[get ] {name}")
    temporary_path = None
    try:
        # A unique partial file in the destination directory makes concurrent
        # runs safe and lets replacement stay atomic on one filesystem.
        with tempfile.NamedTemporaryFile(
                dir=models_dir, prefix=f".{name}.", suffix=".part",
                delete=False) as temporary:
            temporary_path = Path(temporary.name)
        # Download into the partial path with a socket timeout. Streamed writes
        # and an exact byte ceiling prevent an oversized response from filling
        # memory or disk before the checksum can reject it.
        with urllib.request.urlopen(
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised_size = int(content_length)
                except ValueError:
                    advertised_size = None
                if (advertised_size is not None
                        and advertised_size > expected_size):
                    print(
                        f"[fail] {name}: server advertised "
                        f"{advertised_size} bytes; expected {expected_size}"
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
        # Path.replace is an atomic rename when both paths share a filesystem.
        temporary_path.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        print(f"[fail] {name}: {error}")
        return False
    finally:
        # If the rename succeeded this path no longer exists; otherwise remove
        # only this run's partial file, never a previously verified model.
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                # Cleanup trouble should be visible but must not hide the
                # original download or checksum result.
                print(f"[warn] cannot remove partial file {temporary_path}: {error}")
    print(f"[ ok ] {name} sha256={expected_sha}")
    return True


def main() -> int:
    """Entry point: download every model and report an overall exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="re-download files even when they already exist",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=MODELS_DIR,
        help=f"destination directory (default: {MODELS_DIR})",
    )
    arguments = parser.parse_args()
    # Create the models directory on first use.
    try:
        arguments.models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"[fail] cannot create {arguments.models_dir}: {error}")
        return 1
    # Download each model, tracking whether anything failed.
    results = [
        download_one(
            name, url, sha, size, arguments.force, arguments.models_dir
        )
        for name, (url, sha, size) in MODELS.items()
    ]
    # Non-zero exit signals at least one failed download to shell callers.
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

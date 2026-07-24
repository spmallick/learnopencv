#!/usr/bin/env python3
"""Download the ONNX models required by the DNN-based OpenCV trackers.

The classical trackers (MIL, KCF, CSRT) need no model files, so this script
only fetches the assets for TrackerDaSiamRPN, TrackerNano, and TrackerVit.
Every file lands in ``<repo>/Object-Tracking-OpenCV5/models/`` where both the
Python and C++ examples expect to find it.

Usage:
    python3 download_models.py            # download everything that is missing
    python3 download_models.py --force    # re-download even if present
"""

# argparse builds the small command-line interface for this script.
import argparse
# hashlib computes SHA-256 checksums so a truncated or tampered download fails loudly.
import hashlib
# pathlib gives us robust, OS-independent path handling anchored at this file.
from pathlib import Path
# sys provides the exit code used to signal success or failure to callers.
import sys
# urllib.request performs the actual HTTP downloads using only the standard library.
import urllib.request

# Resolve the models directory relative to this script, never the caller's
# current working directory, so the script works from anywhere.
MODELS_DIR = Path(__file__).resolve().parent / "models"

# Each entry: destination filename -> (download URL, expected SHA-256 or None).
# A None checksum means the upstream host does not publish one; the script
# prints the computed hash so it can be pinned after a verified download.
MODELS = {
    # --- TrackerNano (NanoTrack v2): a ~2 MB two-file siamese tracker. ---
    "nanotrack_backbone_sim.onnx": (
        "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/"
        "NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx",
        "530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4",
    ),
    "nanotrack_head_sim.onnx": (
        "https://raw.githubusercontent.com/HonglinChu/SiamTrackers/master/"
        "NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx",
        "0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9",
    ),
    # --- TrackerVit: the transformer tracker from the OpenCV model zoo. ---
    # The zoo stores the file in Git LFS, so we fetch through the media
    # endpoint which serves the real payload instead of the LFS pointer.
    "object_tracking_vittrack_2023sep.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx",
        "2990f0b7cd44d92afa48cd97db6de7be113fc1d9594fddb74e2725c10478e91d",
    ),
    # --- TrackerDaSiamRPN: three files hosted at the URLs documented in the
    # official OpenCV sample (samples/python/tracker.py). ---
    "dasiamrpn_model.onnx": (
        "https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=1",
        None,
    ),
    "dasiamrpn_kernel_r1.onnx": (
        "https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=1",
        None,
    ),
    "dasiamrpn_kernel_cls1.onnx": (
        "https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=1",
        None,
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


def download_one(name: str, url: str, expected_sha: str, force: bool) -> bool:
    """Download a single model file and verify it. Returns True on success."""
    destination = MODELS_DIR / name
    # Skip files that already exist and pass verification unless forced.
    if destination.exists() and not force:
        if expected_sha is None or sha256_of(destination) == expected_sha:
            print(f"[skip] {name} already present")
            return True
        print(f"[warn] {name} exists but fails checksum; re-downloading")
    print(f"[get ] {name}")
    try:
        # A plain urlretrieve is enough; every host serves over HTTPS.
        urllib.request.urlretrieve(url, destination)
    except OSError as error:
        # Report the failure but let the caller decide whether it is fatal;
        # the examples degrade gracefully when a model is missing.
        print(f"[fail] {name}: {error}")
        return False
    # Verify the payload before declaring success.
    actual_sha = sha256_of(destination)
    if expected_sha is not None and actual_sha != expected_sha:
        # Remove the corrupt file so a later run does not trust it.
        destination.unlink()
        print(f"[fail] {name}: checksum mismatch ({actual_sha})")
        return False
    # Print the hash either way so unpinned entries can be pinned later.
    print(f"[ ok ] {name} sha256={actual_sha}")
    return True


def main() -> int:
    """Entry point: download every model and report an overall exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="re-download files even when they already exist",
    )
    arguments = parser.parse_args()
    # Create the models directory on first use.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Download each model, tracking whether anything failed.
    results = [
        download_one(name, url, sha, arguments.force)
        for name, (url, sha) in MODELS.items()
    ]
    # Non-zero exit signals at least one failed download to shell callers.
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

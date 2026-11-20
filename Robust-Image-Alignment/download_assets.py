"""Fetch the article's small, pinned sample set and verify every payload.

Run this file from any working directory. Third-party images remain downloads,
with their original provenance in assets_manifest.json; this script does not
relicense those images. No OpenCV or deep-learning dependency is needed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "assets_manifest.json"


def file_sha256(path: Path) -> str:
    """Hash a file incrementally so verification also works for larger inputs."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_assets(
    output_dir: Path | str | None = None,
    *,
    verify_only: bool = False,
    verbose: bool = True,
) -> Path:
    """Return the sample root after checking all pinned files.

    Existing files with the expected checksum are reused. A missing or changed
    file is fetched again unless verify_only is requested. The manifest stores
    immutable Git commit URLs rather than moving branch URLs.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PROJECT_DIR / manifest["download_root"]
    )
    for entry in manifest["files"]:
        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe relative sample path: {relative_path}")
        destination = root / relative_path
        expected_sha = entry["sha256"]
        if destination.is_file() and file_sha256(destination) == expected_sha:
            continue
        if verify_only:
            raise RuntimeError(f"Missing or checksum-mismatched sample: {destination}")
        if verbose:
            print(f"Downloading {entry['path']}")
        request = urllib.request.Request(
            entry["url"], headers={"User-Agent": "LearnOpenCV-alignment-tutorial"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"Could not download {entry['url']}: {error}") from error
        if len(payload) != entry["bytes"]:
            raise RuntimeError(f"Unexpected byte count for {entry['path']}")
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {entry['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_name(destination.name + ".download")
        temporary_path.write_bytes(payload)
        temporary_path.replace(destination)
    if verbose:
        print(f"Verified {len(manifest['files'])} sample files in {root}")
    return root


def parse_args() -> argparse.Namespace:
    """Keep download and verification commands available outside the notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Override the default assets/downloads directory."
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Check local files without downloading."
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download_assets(arguments.output_dir, verify_only=arguments.verify_only)

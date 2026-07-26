#!/usr/bin/env python3
"""Download the pinned ESPCN x4 model and verify its SHA-256 digest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_URL = (
    "https://raw.githubusercontent.com/fannymonori/TF-ESPCN/"
    "a899033b12cd0400454fb5777600883a9d7e86c3/export/ESPCN_x4.pb"
)
MODEL_SHA256 = "e403f06309229cf36009cd8fb0da032ba7643fae9f15cf94fe562e8edf8fef47"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(output: Path, force: bool = False) -> Path:
    output = Path(output)
    if output.is_file() and not force:
        if sha256(output) == MODEL_SHA256:
            print(f"Model already present and verified: {output}")
            return output
        raise RuntimeError(
            f"{output} exists but has an unexpected checksum. "
            "Use --force to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "LearnOpenCV-model-downloader"})
    temporary_path: Path | None = None
    try:
        with urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=output.parent, prefix=f".{output.name}.", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                while chunk := response.read(1024 * 1024):
                    temporary_file.write(chunk)

        actual_digest = sha256(temporary_path)
        if actual_digest != MODEL_SHA256:
            raise RuntimeError(
                "Downloaded model checksum mismatch: "
                f"expected {MODEL_SHA256}, got {actual_digest}."
            )
        temporary_path.replace(output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Downloaded and verified {output} ({output.stat().st_size} bytes)")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned ESPCN x4 model with checksum verification."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "models" / "ESPCN_x4.pb",
        help="Destination model path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing file, even if its checksum is wrong.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        download_model(args.output, force=args.force)
    except (HTTPError, URLError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

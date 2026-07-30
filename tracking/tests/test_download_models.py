from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import download_models


class FakeResponse:
    def __init__(self, payload: bytes, advertised_size: int | None = None):
        self.payload = payload
        self.offset = 0
        self.headers = {}
        if advertised_size is not None:
            self.headers["Content-Length"] = str(advertised_size)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_manifest_has_all_six_pinned_models() -> None:
    assert set(download_models.MODEL_GROUPS) == {"dasiamrpn", "nano", "vit"}
    entries = {
        filename: values
        for group in download_models.MODEL_GROUPS.values()
        for filename, values in group.items()
    }
    assert len(entries) == 6
    for url, expected_sha, expected_size in entries.values():
        assert url.startswith("https://")
        assert len(expected_sha) == 64
        assert expected_size > 0


def test_valid_download_is_verified_and_replaced(tmp_path: Path) -> None:
    payload = b"verified-model-payload"
    response = FakeResponse(payload, len(payload))
    with mock.patch.object(
        download_models.urllib.request, "urlopen", return_value=response
    ):
        success = download_models.download_one(
            "model.onnx",
            "https://example.invalid/model.onnx",
            _digest(payload),
            len(payload),
            force=True,
            models_dir=tmp_path,
        )
    assert success
    assert (tmp_path / "model.onnx").read_bytes() == payload
    assert not list(tmp_path.glob(".*.part"))


def test_oversized_download_does_not_replace_existing_file(
    tmp_path: Path,
) -> None:
    expected = b"expected"
    destination = tmp_path / "model.onnx"
    destination.write_bytes(b"existing")
    response = FakeResponse(expected + b"extra")
    with mock.patch.object(
        download_models.urllib.request, "urlopen", return_value=response
    ):
        success = download_models.download_one(
            destination.name,
            "https://example.invalid/model.onnx",
            _digest(expected),
            len(expected),
            force=True,
            models_dir=tmp_path,
        )
    assert not success
    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".*.part"))

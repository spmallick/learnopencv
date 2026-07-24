"""Offline regression tests for the atomic model downloader."""

# hashlib provides the exact digest expected by download_one().
import hashlib
# pathlib addresses temporary destinations without current-directory coupling.
from pathlib import Path
# sys lets this test import the downloader from the project root.
import sys
# tempfile isolates every success and failure case.
import tempfile
# unittest provides the repository's standard test framework.
import unittest
# mock replaces only the network response; real file/checksum code still runs.
from unittest import mock

# Resolve and import the real project-root downloader.
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
import download_models  # noqa: E402  (path setup must precede import)


class FakeResponse:
    """Small context-managed response that streams a fixed byte payload."""

    def __init__(self, payload, advertised_size=None):
        # Preserve the payload and a read cursor so chunked reads are realistic.
        self.payload = payload
        self.offset = 0
        # Omit Content-Length when requested to exercise the streaming ceiling.
        self.headers = {}
        if advertised_size is not None:
            self.headers["Content-Length"] = str(advertised_size)

    def __enter__(self):
        """Return the response just like urllib's HTTPResponse context manager."""
        return self

    def __exit__(self, _exception_type, _exception, _traceback):
        """Do not suppress exceptions raised by the downloader."""
        return False

    def read(self, size):
        """Return at most ``size`` bytes and then signal EOF with ``b''``."""
        chunk = self.payload[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def digest(payload):
    """Return the SHA-256 digest used in the downloader's pinned metadata."""
    return hashlib.sha256(payload).hexdigest()


class DownloadOneTests(unittest.TestCase):
    """Exercise success, bounds, integrity, timeout, and cleanup behavior."""

    def _run_download(
            self, directory, payload, expected_payload,
            advertised_size=None, force=True):
        """Call the real downloader with a deterministic mocked response."""
        response = FakeResponse(payload, advertised_size)
        with mock.patch.object(
                download_models.urllib.request, "urlopen",
                return_value=response):
            return download_models.download_one(
                "model.onnx",
                "https://example.invalid/model.onnx",
                digest(expected_payload),
                len(expected_payload),
                force,
                Path(directory),
            )

    def _assert_no_partials(self, directory):
        """A completed or rejected request must leave no temporary files."""
        partials = list(Path(directory).glob(".*.part"))
        self.assertEqual(partials, [])

    def test_valid_payload_replaces_destination_atomically(self):
        # The happy path should expose exactly the verified final file.
        payload = b"verified-model-payload"
        with tempfile.TemporaryDirectory() as directory:
            success = self._run_download(
                directory, payload, payload, advertised_size=len(payload)
            )
            destination = Path(directory) / "model.onnx"
            self.assertTrue(success)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(set(Path(directory).iterdir()), {destination})
            self._assert_no_partials(directory)

    def test_oversized_stream_preserves_existing_destination(self):
        # Without Content-Length, the streaming byte ceiling must still stop an
        # oversized body before it can replace the prior file.
        expected = b"expected"
        existing = b"previously-verified"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.onnx"
            destination.write_bytes(existing)
            success = self._run_download(
                directory, expected + b"extra", expected,
                advertised_size=None,
            )
            self.assertFalse(success)
            self.assertEqual(destination.read_bytes(), existing)
            self._assert_no_partials(directory)

    def test_undersized_stream_preserves_existing_destination(self):
        # A clean EOF is not success unless the exact pinned byte count arrived.
        expected = b"complete-model"
        existing = b"previously-verified"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.onnx"
            destination.write_bytes(existing)
            success = self._run_download(
                directory, expected[:-1], expected,
                advertised_size=len(expected) - 1,
            )
            self.assertFalse(success)
            self.assertEqual(destination.read_bytes(), existing)
            self._assert_no_partials(directory)

    def test_bad_checksum_preserves_existing_destination(self):
        # Equal byte counts are insufficient when the pinned digest disagrees.
        expected = b"expected-model"
        corrupted = b"corruptd-model"
        self.assertEqual(len(corrupted), len(expected))
        existing = b"previously-verified"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.onnx"
            destination.write_bytes(existing)
            success = self._run_download(
                directory, corrupted, expected,
                advertised_size=len(expected),
            )
            self.assertFalse(success)
            self.assertEqual(destination.read_bytes(), existing)
            self._assert_no_partials(directory)

    def test_timeout_preserves_existing_destination_and_cleans_partial(self):
        # Network timeouts are ordinary failures: no traceback, replacement,
        # or abandoned partial file should result.
        existing = b"previously-verified"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.onnx"
            destination.write_bytes(existing)
            with mock.patch.object(
                    download_models.urllib.request, "urlopen",
                    side_effect=TimeoutError("timed out")):
                success = download_models.download_one(
                    "model.onnx",
                    "https://example.invalid/model.onnx",
                    digest(b"expected"),
                    len(b"expected"),
                    True,
                    Path(directory),
                )
            self.assertFalse(success)
            self.assertEqual(destination.read_bytes(), existing)
            self._assert_no_partials(directory)


if __name__ == "__main__":
    unittest.main()

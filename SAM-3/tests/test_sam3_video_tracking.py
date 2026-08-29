from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam3_video_tracking import (
    overlay_tracking_outputs,
    stream_tracking_outputs,
    track_video,
)


class FakePredictor:
    def __init__(self, frame_count: int, height: int, width: int) -> None:
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.requests: list[dict] = []

    def _outputs(self, frame_index: int) -> dict[str, np.ndarray]:
        mask = np.zeros((1, self.height, self.width), dtype=bool)
        x_start = 8 + frame_index
        mask[:, 10:30, x_start : x_start + 20] = True
        return {
            "out_obj_ids": np.array([7], dtype=np.int64),
            "out_probs": np.array([0.91], dtype=np.float32),
            "out_binary_masks": mask,
        }

    def handle_request(self, request: dict) -> dict:
        self.requests.append(request.copy())
        if request["type"] == "start_session":
            return {"session_id": "test-session"}
        if request["type"] == "add_prompt":
            return {
                "frame_index": request["frame_index"],
                "outputs": self._outputs(request["frame_index"]),
            }
        if request["type"] == "close_session":
            return {"is_success": True}
        raise AssertionError(f"Unexpected request: {request}")

    def handle_stream_request(self, request: dict):
        self.requests.append(request.copy())
        for frame_index in range(self.frame_count):
            yield {
                "frame_index": frame_index,
                "outputs": self._outputs(frame_index),
            }


class OverlayTests(unittest.TestCase):
    def test_overlay_changes_only_mask_region_except_labels(self) -> None:
        frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((1, 48, 64), dtype=bool)
        mask[:, 16:32, 20:44] = True
        outputs = {
            "out_obj_ids": np.array([1]),
            "out_binary_masks": mask,
        }

        rendered = overlay_tracking_outputs(frame, outputs, alpha=0.5, blur_kernel=1)

        np.testing.assert_array_equal(rendered[40, 60], frame[40, 60])
        self.assertFalse(np.array_equal(rendered[24, 32], frame[24, 32]))

    def test_empty_outputs_leave_frame_unchanged(self) -> None:
        frame = np.full((20, 30, 3), 42, dtype=np.uint8)
        rendered = overlay_tracking_outputs(
            frame,
            {"out_obj_ids": np.array([]), "out_binary_masks": np.array([])},
        )
        np.testing.assert_array_equal(rendered, frame)

    def test_mismatched_object_and_mask_counts_fail(self) -> None:
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        outputs = {
            "out_obj_ids": np.array([1, 2]),
            "out_binary_masks": np.zeros((1, 20, 30), dtype=bool),
        }
        with self.assertRaisesRegex(ValueError, "different object-ID and mask counts"):
            overlay_tracking_outputs(frame, outputs)


class PipelineTests(unittest.TestCase):
    def test_propagated_frame_zero_replaces_immediate_prompt_output(self) -> None:
        class HotStartFilteringPredictor(FakePredictor):
            def handle_request(self, request: dict) -> dict:
                response = super().handle_request(request)
                if request["type"] == "add_prompt":
                    response["outputs"]["out_obj_ids"] = np.array([99])
                return response

        predictor = HotStartFilteringPredictor(frame_count=2, height=10, width=10)
        responses = list(
            stream_tracking_outputs(predictor, Path("unused.mp4"), "person")
        )

        self.assertEqual([response["frame_index"] for response in responses], [0, 1])
        self.assertEqual(int(responses[0]["outputs"]["out_obj_ids"][0]), 7)

    def test_pinned_upstream_start_session_regression_is_adapted(self) -> None:
        class MultiplexModel:
            def __init__(self) -> None:
                self.resource_path: str | None = None

            # This intentionally matches the incompatible SAM 3.1 signature at
            # Meta commit 660a5e9: offload_state_to_cpu is not accepted.
            def init_state(self, resource_path: str, offload_video_to_cpu: bool = False):
                self.resource_path = resource_path
                return {"offload_video_to_cpu": offload_video_to_cpu}

        class RegressedBasePredictor(FakePredictor):
            def __init__(self) -> None:
                super().__init__(frame_count=1, height=10, width=10)
                self.model = MultiplexModel()

            def handle_request(self, request: dict) -> dict:
                if request["type"] == "start_session":
                    self.requests.append(request.copy())
                    self.model.init_state(
                        resource_path=request["resource_path"],
                        offload_video_to_cpu=False,
                        offload_state_to_cpu=False,
                    )
                    return {"session_id": "test-session"}
                return super().handle_request(request)

        predictor = RegressedBasePredictor()
        responses = list(
            stream_tracking_outputs(predictor, Path("input.mp4"), "person")
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(predictor.model.resource_path, "input.mp4")
        self.assertNotIn("init_state", predictor.model.__dict__)

    def test_fake_predictor_session_renders_decodable_video(self) -> None:
        frame_count, width, height = 4, 64, 48
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            input_path = temp / "input.avi"
            output_path = temp / "output.avi"

            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                8.0,
                (width, height),
            )
            self.assertTrue(writer.isOpened())
            for index in range(frame_count):
                writer.write(np.full((height, width, 3), 40 + index, dtype=np.uint8))
            writer.release()

            predictor = FakePredictor(frame_count, height, width)
            summary = track_video(
                predictor,
                input_path,
                output_path,
                "person",
                codec="MJPG",
                blur_kernel=1,
                preserve_audio=False,
            )

            self.assertEqual(summary["tracked_frames"], frame_count)
            self.assertEqual(summary["rendered_frames"], frame_count)
            self.assertEqual(
                [request["type"] for request in predictor.requests],
                ["start_session", "add_prompt", "propagate_in_video", "close_session"],
            )
            self.assertEqual(predictor.requests[1]["text"], "person")

            capture = cv2.VideoCapture(str(output_path))
            decoded = 0
            changed_inside_mask = False
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                changed_inside_mask |= bool(np.max(frame[20, 16]) > 45)
            capture.release()

            self.assertEqual(decoded, frame_count)
            self.assertTrue(changed_inside_mask)

    def test_session_closes_when_propagation_fails(self) -> None:
        class FailingPredictor(FakePredictor):
            def handle_stream_request(self, request: dict):
                self.requests.append(request.copy())
                raise RuntimeError("synthetic propagation failure")
                yield  # pragma: no cover - keeps this method a generator

        predictor = FailingPredictor(frame_count=1, height=10, width=10)
        with self.assertRaisesRegex(RuntimeError, "synthetic propagation failure"):
            list(stream_tracking_outputs(predictor, Path("unused.mp4"), "person"))
        self.assertEqual(predictor.requests[-1]["type"], "close_session")

    def test_render_failure_removes_partial_output_and_closes_session(self) -> None:
        class FailingPredictor(FakePredictor):
            def handle_stream_request(self, request: dict):
                self.requests.append(request.copy())
                raise RuntimeError("synthetic propagation failure")
                yield  # pragma: no cover - keeps this method a generator

        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            input_path = temp / "input.avi"
            output_path = temp / "output.avi"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                8.0,
                (64, 48),
            )
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()

            predictor = FailingPredictor(frame_count=1, height=48, width=64)
            with self.assertRaisesRegex(RuntimeError, "synthetic propagation failure"):
                track_video(
                    predictor,
                    input_path,
                    output_path,
                    "person",
                    codec="MJPG",
                    preserve_audio=False,
                )

            self.assertEqual(predictor.requests[-1]["type"], "close_session")
            self.assertFalse(output_path.exists())
            self.assertEqual(list(temp.glob(".output.video-only.*.avi")), [])

    def test_output_temp_name_cannot_overwrite_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            input_path = temp / "result.video-only.avi"
            output_path = temp / "result.avi"
            writer = cv2.VideoWriter(
                str(input_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                8.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for index in range(3):
                writer.write(np.full((48, 64, 3), 20 + index, dtype=np.uint8))
            writer.release()
            source_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()

            predictor = FakePredictor(frame_count=3, height=48, width=64)
            track_video(
                predictor,
                input_path,
                output_path,
                "person",
                codec="MJPG",
                preserve_audio=False,
            )

            self.assertEqual(
                hashlib.sha256(input_path.read_bytes()).hexdigest(), source_sha256
            )
            self.assertTrue(output_path.is_file())
            self.assertEqual(output_path.stat().st_mode & 0o777, 0o644)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_source_audio_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            input_path = temp / "input-with-audio.mp4"
            output_path = temp / "output-with-audio.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x48:r=8:d=1.0",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=1000:duration=0.25",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    str(input_path),
                ],
                check=True,
            )

            predictor = FakePredictor(frame_count=8, height=48, width=64)
            track_video(
                predictor,
                input_path,
                output_path,
                "person",
                preserve_audio=True,
            )

            probe = subprocess.run(
                [
                    shutil.which("ffprobe"),
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(set(probe.stdout.split()), {"video", "audio"})

            capture = cv2.VideoCapture(str(output_path))
            decoded_frames = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                decoded_frames += 1
            capture.release()
            self.assertEqual(decoded_frames, 8)


if __name__ == "__main__":
    unittest.main()

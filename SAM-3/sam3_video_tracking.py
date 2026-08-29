#!/usr/bin/env python3
"""Stateful SAM 3.1 text-prompted video tracking with OpenCV rendering.

The SAM 3.1 predictor owns temporal state and propagates object identities.
OpenCV is used only to decode the source video and render the returned masks.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


# BGR colors selected for strong contrast on common video backgrounds.
_OBJECT_COLORS = (
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (155, 89, 182),
    (241, 196, 15),
    (26, 188, 156),
    (230, 126, 34),
    (149, 165, 166),
)


def _temporary_output_path(output_path: Path, label: str) -> Path:
    """Reserve a unique temporary file beside the requested output."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.{label}.",
        suffix=output_path.suffix,
    )
    os.close(descriptor)
    return Path(temporary_name)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy-compatible value or PyTorch tensor without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _binary_mask(mask: Any, frame_height: int, frame_width: int) -> np.ndarray:
    """Return one SAM mask as a full-resolution boolean array."""
    array = np.squeeze(_as_numpy(mask))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask after squeezing, got {array.shape}")
    if array.shape != (frame_height, frame_width):
        array = cv2.resize(
            array.astype(np.float32),
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return array.astype(np.float32) > 0.5


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return inclusive XYXY bounds for a nonempty mask."""
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())


def overlay_tracking_outputs(
    frame_bgr: np.ndarray,
    outputs: Mapping[str, Any] | None,
    *,
    alpha: float = 0.45,
    blur_kernel: int = 11,
) -> np.ndarray:
    """Blend per-object SAM masks and stable IDs onto one OpenCV frame.

    A small Gaussian blur softens each mask boundary, so blending is confined
    to the mask and a narrow edge band rather than tinting the complete frame.
    """
    if outputs is None:
        return frame_bgr.copy()
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if blur_kernel < 1 or blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be a positive odd integer")

    object_ids = _as_numpy(outputs.get("out_obj_ids", [])).reshape(-1)
    raw_masks = _as_numpy(outputs.get("out_binary_masks", []))
    probabilities = _as_numpy(outputs.get("out_probs", [])).reshape(-1)

    if object_ids.size == 0 or raw_masks.size == 0:
        return frame_bgr.copy()
    if raw_masks.ndim == 2:
        raw_masks = raw_masks[np.newaxis, ...]
    if len(raw_masks) != len(object_ids):
        raise ValueError(
            "SAM output has different object-ID and mask counts: "
            f"{len(object_ids)} IDs versus {len(raw_masks)} masks"
        )

    height, width = frame_bgr.shape[:2]
    rendered = frame_bgr.astype(np.float32)
    labels: list[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]] = []

    for index, (object_id, raw_mask) in enumerate(zip(object_ids, raw_masks)):
        mask = _binary_mask(raw_mask, height, width)
        bounds = _mask_bounds(mask)
        if bounds is None:
            continue

        color = _OBJECT_COLORS[int(object_id) % len(_OBJECT_COLORS)]
        soft_mask = mask.astype(np.float32)
        if blur_kernel > 1:
            soft_mask = cv2.GaussianBlur(soft_mask, (blur_kernel, blur_kernel), 0)
        local_alpha = np.clip(soft_mask * alpha, 0.0, 1.0)[..., np.newaxis]
        rendered = rendered * (1.0 - local_alpha) + np.asarray(color) * local_alpha

        label = f"ID {int(object_id)}"
        if index < len(probabilities):
            label += f"  {float(probabilities[index]):.2f}"
        labels.append((bounds, label, color))

    rendered = np.clip(rendered, 0, 255).astype(np.uint8)
    for (x_min, y_min, x_max, y_max), label, color in labels:
        cv2.rectangle(rendered, (x_min, y_min), (x_max, y_max), color, 2)
        text_origin = (x_min, max(18, y_min - 7))
        cv2.putText(
            rendered,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return rendered


def stream_tracking_outputs(
    predictor: Any,
    video_path: Path,
    prompt: str,
    *,
    output_threshold: float = 0.5,
) -> Iterable[Mapping[str, Any]]:
    """Yield ordered SAM outputs while keeping only the current masks in memory."""
    session_id: str | None = None
    try:
        start_response = _handle_start_session(
            predictor,
            {
                "type": "start_session",
                "resource_path": str(video_path),
            },
        )
        session_id = start_response["session_id"]

        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": prompt,
                "output_prob_thresh": output_threshold,
            }
        )

        stream: Iterable[Mapping[str, Any]] = predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "output_prob_thresh": output_threshold,
            }
        )
        for response in stream:
            # Propagation output is canonical even for frame zero: SAM 3.1 can
            # remove unconfirmed hot-start objects using evidence from later frames.
            yield response
    finally:
        if session_id is not None:
            predictor.handle_request(
                {
                    "type": "close_session",
                    "session_id": session_id,
                }
            )


def _handle_start_session(
    predictor: Any,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Start a session while tolerating Meta SAM 3.1 issue #544.

    Meta commit 660a5e9 forwards ``offload_state_to_cpu`` from the base
    predictor even though the multiplex model's ``init_state`` method does not
    accept it. Temporarily filtering that one unsupported keyword preserves the
    upstream request API and becomes a no-op once Meta's implementation accepts
    the argument.
    """
    model = getattr(predictor, "model", None)
    init_state = getattr(model, "init_state", None)
    if init_state is None:
        return predictor.handle_request(dict(request))

    try:
        parameters = inspect.signature(init_state).parameters.values()
    except (TypeError, ValueError):
        return predictor.handle_request(dict(request))

    accepts_offload_state = any(
        parameter.name == "offload_state_to_cpu"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_offload_state:
        return predictor.handle_request(dict(request))

    def compatible_init_state(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("offload_state_to_cpu", None)
        return init_state(*args, **kwargs)

    model_namespace = getattr(model, "__dict__", {})
    had_instance_override = "init_state" in model_namespace
    previous_instance_value = model_namespace.get("init_state")
    try:
        setattr(model, "init_state", compatible_init_state)
    except (AttributeError, TypeError) as error:
        raise RuntimeError(
            "This SAM 3.1 checkout has Meta issue #544 and its model cannot be "
            "adapted safely. Use the pinned companion instructions or an official "
            "SAM revision where the multiplex start-session fix is merged."
        ) from error

    try:
        return predictor.handle_request(dict(request))
    finally:
        if had_instance_override:
            setattr(model, "init_state", previous_instance_value)
        else:
            delattr(model, "init_state")


def render_tracking_video(
    video_path: Path,
    output_path: Path,
    output_stream: Iterable[Mapping[str, Any]],
    *,
    alpha: float = 0.45,
    blur_kernel: int = 11,
    codec: str = "mp4v",
    preserve_audio: bool = True,
) -> tuple[int, int]:
    """Render streamed SAM outputs and return rendered/tracked frame counts."""
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")

    video_path = video_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if video_path == output_path:
        raise ValueError("Input and output video paths must be different")
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the input video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video reports invalid FPS or frame dimensions")

    partial_path = _temporary_output_path(output_path, "video-only")
    writer = cv2.VideoWriter(
        str(partial_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"OpenCV could not create the output video: {partial_path}")

    frame_index = 0
    tracked_frames = 0
    render_failed = False
    try:
        for response in output_stream:
            response_frame = int(response["frame_index"])
            if response_frame < frame_index:
                raise RuntimeError(
                    "SAM propagation returned frames out of order: "
                    f"received {response_frame} after {frame_index - 1}"
                )

            while frame_index < response_frame:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"SAM returned frame {response_frame}, beyond the decoded video"
                    )
                writer.write(frame)
                frame_index += 1

            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"SAM returned frame {response_frame}, beyond the decoded video"
                )
            rendered = overlay_tracking_outputs(
                frame,
                response["outputs"],
                alpha=alpha,
                blur_kernel=blur_kernel,
            )
            writer.write(rendered)
            frame_index += 1
            tracked_frames += 1

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_index += 1
    except Exception:
        render_failed = True
        raise
    finally:
        capture.release()
        writer.release()
        if render_failed:
            partial_path.unlink(missing_ok=True)

    if frame_index == 0:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError("Input video contained no decodable frames")

    try:
        if preserve_audio and _source_has_audio(video_path):
            _mux_source_audio(partial_path, video_path, output_path)
            partial_path.unlink(missing_ok=True)
        else:
            partial_path.replace(output_path)
        output_path.chmod(0o644)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise
    return frame_index, tracked_frames


def _source_has_audio(video_path: Path) -> bool:
    """Use FFprobe to determine whether the source contains an audio stream."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError(
            "FFprobe is required to preserve source audio; install FFmpeg or pass --no-audio"
        )
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def _mux_source_audio(video_only_path: Path, source_path: Path, output_path: Path) -> None:
    """Copy rendered video and transcode the source audio into the final container."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required to preserve source audio; install FFmpeg or pass --no-audio"
        )

    muxed_path = _temporary_output_path(output_path, "muxed")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_only_path),
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-af",
        "apad",
        "-shortest",
        str(muxed_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        muxed_path.replace(output_path)
    except Exception as error:
        muxed_path.unlink(missing_ok=True)
        detail = getattr(error, "stderr", "") or "unknown FFmpeg error"
        detail = str(detail).strip()
        raise RuntimeError(f"Could not preserve source audio: {detail}") from error


def build_sam31_predictor(
    *,
    checkpoint_path: Path | None = None,
    max_num_objects: int = 16,
    multiplex_count: int = 16,
    compile_model: bool = False,
    use_fa3: bool = False,
) -> Any:
    """Build Meta's current SAM 3.1 Object Multiplex video predictor."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3.1 video inference requires a CUDA-capable GPU")

    from sam3.model_builder import build_sam3_multiplex_video_predictor

    return build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        compile=compile_model,
        warm_up=compile_model,
        use_fa3=use_fa3,
    )


def track_video(
    predictor: Any,
    video_path: Path,
    output_path: Path,
    prompt: str,
    *,
    output_threshold: float = 0.5,
    alpha: float = 0.45,
    blur_kernel: int = 11,
    codec: str = "mp4v",
    preserve_audio: bool = True,
) -> dict[str, int | str]:
    """Run stateful tracking and render a video using an existing predictor."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must contain one noun phrase")
    if not 0.0 <= output_threshold <= 1.0:
        raise ValueError("output_threshold must be between 0 and 1")

    output_stream = stream_tracking_outputs(
        predictor,
        video_path,
        prompt,
        output_threshold=output_threshold,
    )
    try:
        rendered_frames, tracked_frames = render_tracking_video(
            video_path,
            output_path,
            output_stream,
            alpha=alpha,
            blur_kernel=blur_kernel,
            codec=codec,
            preserve_audio=preserve_audio,
        )
    finally:
        close_stream = getattr(output_stream, "close", None)
        if close_stream is not None:
            close_stream()
    return {
        "prompt": prompt,
        "tracked_frames": tracked_frames,
        "rendered_frames": rendered_frames,
        "output_path": str(output_path.expanduser().resolve()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track every instance of one text-described concept with SAM 3.1."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 video")
    parser.add_argument("--output", type=Path, required=True, help="Rendered output video")
    parser.add_argument(
        "--prompt",
        required=True,
        help='One noun phrase, for example "person wearing a red shirt"',
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument(
        "--multiplex-count",
        type=int,
        default=16,
        help=(
            "Object Multiplex bucket width (the public SAM 3.1 checkpoint uses 16; "
            "other values require a compatible checkpoint)"
        ),
    )
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--fa3",
        action="store_true",
        dest="use_fa3",
        help="Enable optional FlashAttention 3 kernels on a compatible system",
    )
    parser.set_defaults(use_fa3=False)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--blur-kernel", type=int, default=11)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--no-audio",
        action="store_false",
        dest="preserve_audio",
        help="Do not copy the source audio into the rendered output",
    )
    parser.set_defaults(preserve_audio=True)
    args = parser.parse_args(argv)

    if args.max_objects < 1:
        parser.error("--max-objects must be positive")
    if args.multiplex_count < 1:
        parser.error("--multiplex-count must be positive")
    if args.multiplex_count > args.max_objects:
        parser.error("--multiplex-count cannot exceed --max-objects")
    if args.multiplex_count != 16 and args.checkpoint is None:
        parser.error(
            "the public SAM 3.1 checkpoint requires --multiplex-count 16; "
            "provide a compatible --checkpoint for another value"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictor = build_sam31_predictor(
        checkpoint_path=args.checkpoint,
        max_num_objects=args.max_objects,
        multiplex_count=args.multiplex_count,
        compile_model=args.compile_model,
        use_fa3=args.use_fa3,
    )
    summary = track_video(
        predictor,
        args.video,
        args.output,
        args.prompt,
        output_threshold=args.threshold,
        alpha=args.alpha,
        blur_kernel=args.blur_kernel,
        codec=args.codec,
        preserve_audio=args.preserve_audio,
    )
    print(
        "Saved {rendered_frames} frames to {output_path} "
        "({tracked_frames} frames carried SAM outputs).".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

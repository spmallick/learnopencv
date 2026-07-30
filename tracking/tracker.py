#!/usr/bin/env python3
"""Track one object in a video with OpenCV's modern single-object tracker API."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2


# Resolve the default model folder beside this script, independent of cwd.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"
# Classic trackers construct without external neural-network weights.
CLASSIC_TRACKER_NAMES = ("MIL", "CSRT", "KCF")
# Each model-backed tracker must receive exactly the files its Params object uses.
MODEL_TRACKER_FILES = {
    "DASIAMRPN": (
        "dasiamrpn_model.onnx",
        "dasiamrpn_kernel_cls1.onnx",
        "dasiamrpn_kernel_r1.onnx",
    ),
    "NANO": (
        "nanotrack_backbone_sim.onnx",
        "nanotrack_head_sim.onnx",
    ),
    "VIT": ("object_tracking_vittrack_2023sep.onnx",),
}
# Class names are kept as data because Python bindings expose two layouts.
MODEL_TRACKER_CLASSES = {
    "DASIAMRPN": "TrackerDaSiamRPN",
    "NANO": "TrackerNano",
    "VIT": "TrackerVit",
}
# Accept the longer names readers commonly use while keeping a compact CLI.
TRACKER_ALIASES = {"NANOTRACK": "NANO", "VITTRACK": "VIT"}
TRACKER_NAMES = (*CLASSIC_TRACKER_NAMES, *MODEL_TRACKER_FILES)


def _normalized_tracker_name(name: str) -> str:
    """Return the canonical command-line name used in summaries and errors."""
    normalized = name.upper()
    return TRACKER_ALIASES.get(normalized, normalized)


def _tracker_creator(class_name: str) -> Any | None:
    """Resolve either spelling used by OpenCV's Python tracker bindings."""
    flat_creator = getattr(cv2, f"{class_name}_create", None)
    if flat_creator is not None:
        return flat_creator
    tracker_class = getattr(cv2, class_name, None)
    if tracker_class is not None:
        return getattr(tracker_class, "create", None)
    return None


def _tracker_params(class_name: str) -> Any | None:
    """Create the Params object exposed by either binding layout."""
    params_class = getattr(cv2, f"{class_name}_Params", None)
    if params_class is None:
        tracker_class = getattr(cv2, class_name, None)
        params_class = (
            getattr(tracker_class, "Params", None)
            if tracker_class is not None
            else None
        )
    return params_class() if params_class is not None else None


def _create_model_tracker(name: str, models_dir: Path) -> Any:
    """Validate model assets, configure the DNN backend, and create a tracker."""
    required_files = MODEL_TRACKER_FILES[name]
    # Check the complete set before OpenCV emits a less actionable load error.
    missing = [
        models_dir / filename
        for filename in required_files
        if not (models_dir / filename).is_file()
    ]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"{name} requires model files in {models_dir}; missing: "
            f"{missing_names}. Run: python download_models.py "
            f"--tracker {name.lower()}"
        )

    # Resolve both factory spellings used across wheel and source builds.
    class_name = MODEL_TRACKER_CLASSES[name]
    creator = _tracker_creator(class_name)
    params = _tracker_params(class_name)
    if creator is None or params is None:
        raise RuntimeError(
            f"{name} is unavailable in this OpenCV build. Use OpenCV 4.9 or "
            "newer with the video and DNN modules."
        )
    # Tracker classes can be present in a build whose DNN module was disabled.
    dnn = getattr(cv2, "dnn", None)
    if dnn is None:
        raise RuntimeError(
            f"{name} requires an OpenCV build that includes the DNN module."
        )

    # Populate the differently named path fields in each OpenCV Params class.
    if name == "DASIAMRPN":
        params.model = str(models_dir / required_files[0])
        params.kernel_cls1 = str(models_dir / required_files[1])
        params.kernel_r1 = str(models_dir / required_files[2])
    elif name == "NANO":
        params.backbone = str(models_dir / required_files[0])
        params.neckhead = str(models_dir / required_files[1])
    else:
        params.net = str(models_dir / required_files[0])

    # Explicit CPU settings provide the same documented path in OpenCV 4 and 5.
    params.backend = dnn.DNN_BACKEND_OPENCV
    params.target = dnn.DNN_TARGET_CPU
    try:
        return creator(params)
    except cv2.error as exc:
        raise RuntimeError(
            f"OpenCV could not create {name}. Confirm that this build includes "
            "the DNN module and that download_models.py verified every model."
        ) from exc


def create_tracker(name: str, models_dir: Path = DEFAULT_MODELS_DIR) -> Any:
    """Create a classic or model-backed tracker exposed by this OpenCV build."""
    normalized = _normalized_tracker_name(name)
    if normalized not in TRACKER_NAMES:
        raise ValueError(
            f"Unsupported tracker '{name}'. Choose one of: {', '.join(TRACKER_NAMES)}"
        )

    # Neural trackers need explicit, checksum-verified model paths.
    if normalized in MODEL_TRACKER_FILES:
        return _create_model_tracker(normalized, Path(models_dir))

    # Classic factories may be in the main namespace or OpenCV 4's legacy one.
    factory_name = f"Tracker{normalized}_create"
    for namespace in (cv2, getattr(cv2, "legacy", None)):
        if namespace is None:
            continue
        if hasattr(namespace, factory_name):
            return getattr(namespace, factory_name)()
        tracker_class = getattr(namespace, f"Tracker{normalized}", None)
        if tracker_class is not None and hasattr(tracker_class, "create"):
            return tracker_class.create()

    if normalized in {"CSRT", "KCF"}:
        raise RuntimeError(
            f"{normalized} is unavailable in this build. It is a contrib tracker "
            "when exposed and was absent from the exact OpenCV 5.0 builds tested "
            "here; use MIL for the verified cross-version example."
        )
    raise RuntimeError(
        "MIL is unavailable. Install an OpenCV build that includes the video "
        "module's tracker API."
    )


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    """Parse x,y,width,height and reject non-positive boxes."""
    try:
        box = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "The bounding box must contain four integers: x,y,width,height"
        ) from exc
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise argparse.ArgumentTypeError(
            "The bounding box must be x,y,width,height with positive dimensions"
        )
    return box


def _checked_bbox(
    bbox: tuple[int, int, int, int], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > frame_width
        or y + height > frame_height
    ):
        raise ValueError(
            f"Bounding box {bbox} lies outside the {frame_width}x{frame_height} frame"
        )
    return bbox


def _open_writer(
    path: Path, fps: float, frame_size: tuple[int, int]
) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "mp4v" if path.suffix.lower() == ".mp4" else "MJPG"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {path}")
    return writer


def run_tracking(
    input_path: Path,
    bbox: tuple[int, int, int, int] | None,
    *,
    tracker_name: str = "MIL",
    models_dir: Path = DEFAULT_MODELS_DIR,
    output_path: Path | None = None,
    snapshot_path: Path | None = None,
    max_frames: int | None = None,
    display: bool = False,
    select_roi: bool = False,
) -> dict[str, Any]:
    """Run tracking and return machine-readable measurements."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    writer: cv2.VideoWriter | None = None
    last_annotated = None
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read the first frame: {input_path}")

        frame_height, frame_width = frame.shape[:2]
        if select_roi:
            selected = cv2.selectROI("Select object", frame, False, False)
            cv2.destroyWindow("Select object")
            bbox = tuple(int(value) for value in selected)
        if bbox is None:
            raise ValueError("Provide --bbox or use --select-roi")
        bbox = _checked_bbox(bbox, frame_width, frame_height)

        tracker = create_tracker(tracker_name, models_dir)
        init_result = tracker.init(frame, bbox)
        if init_result is False:
            raise RuntimeError("Tracker initialization failed")

        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        if output_path is not None:
            writer = _open_writer(
                output_path, source_fps, (frame_width, frame_height)
            )

        frames_processed = 0
        successful_updates = 0
        elapsed_total = 0.0
        last_bbox = bbox

        while max_frames is None or frames_processed < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            start = perf_counter()
            found, updated_bbox = tracker.update(frame)
            elapsed = perf_counter() - start
            elapsed_total += elapsed
            frames_processed += 1

            if found:
                last_bbox = tuple(int(round(value)) for value in updated_bbox)
                x, y, width, height = last_bbox
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (255, 0, 0),
                    2,
                )
                successful_updates += 1
            else:
                cv2.putText(
                    frame,
                    "Tracking failure detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"{tracker_name.upper()} | {instantaneous_fps:.1f} FPS",
                (20, frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 170, 50),
                2,
            )
            last_annotated = frame

            if writer is not None:
                writer.write(frame)
            if display:
                cv2.imshow("Object tracking", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

        if frames_processed == 0:
            raise RuntimeError("The input contains no frames after initialization")
        if snapshot_path is not None and last_annotated is not None:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(snapshot_path), last_annotated):
                raise RuntimeError(f"Could not write snapshot: {snapshot_path}")

        return {
            "tracker": _normalized_tracker_name(tracker_name),
            "frames_processed": frames_processed,
            "successful_updates": successful_updates,
            "success_rate": successful_updates / frames_processed,
            "last_bbox": list(last_bbox),
            "frame_size": [frame_width, frame_height],
            "source_fps": source_fps,
            "mean_tracking_ms": 1000.0 * elapsed_total / frames_processed,
        }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display or select_roi:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "videos" / "chaplin.mp4",
    )
    parser.add_argument(
        "--tracker",
        choices=TRACKER_NAMES,
        default="MIL",
        help=(
            "MIL/CSRT/KCF are model-free; DASIAMRPN/NANO/VIT use ONNX files"
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"verified ONNX directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=(287, 23, 86, 320),
        help="Initial x,y,width,height (default: 287,23,86,320)",
    )
    parser.add_argument("--select-roi", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run_tracking(
            args.input,
            args.bbox,
            tracker_name=args.tracker,
            models_dir=args.models_dir,
            output_path=args.output,
            snapshot_path=args.snapshot,
            max_frames=args.max_frames,
            display=args.display,
            select_roi=args.select_roi,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

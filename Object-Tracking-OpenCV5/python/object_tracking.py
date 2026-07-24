#!/usr/bin/env python3
"""Single-object tracking with OpenCV 5's six modern, non-legacy trackers.

One command-line application drives all six trackers:

    classical (no model files):  MIL, KCF, CSRT
    deep-learning (ONNX files):  DaSiamRPN, NanoTrack, TrackerVit

The same source runs unchanged on OpenCV 4.x (4.9+) and OpenCV 5.x, which is
exactly the compatibility contract the accompanying LearnOpenCV article
explains. KCF and CSRT come from opencv_contrib, so they appear only when the
contrib build (for example the ``opencv-contrib-python`` wheel) is installed.

Examples:
    python3 object_tracking.py --list-trackers
    python3 object_tracking.py --tracker vittrack --input video.mp4
    python3 object_tracking.py --tracker csrt --input 0            # webcam
    python3 object_tracking.py --tracker mil --validate --no-display
"""

# argparse implements the command-line interface shared by all trackers.
import argparse
# json serializes the run metrics so tests and benchmarks can consume them.
import json
# pathlib anchors every bundled asset at this file, not the caller's cwd.
from pathlib import Path
# sys supplies exit codes and access to the interpreter for error reporting.
import sys

# NumPy generates the deterministic synthetic validation video.
import numpy as np

# OpenCV provides the tracker implementations, video I/O, and drawing calls.
import cv2

# The models directory sits one level above this script and is shared with
# the C++ example, so both languages read identical ONNX files.
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Geometry of the synthetic validation clip. Kept small so validation runs in
# a couple of seconds yet long enough to expose drift.
SYNTH_SIZE = (640, 360)      # frame width and height in pixels
SYNTH_FRAMES = 80            # number of frames in the clip
SYNTH_TARGET = 64            # side length of the moving square target
SYNTH_SEED = 7               # fixed seed makes every run byte-identical

# Validation passes when the tracked box stays this close to ground truth.
# The thresholds are intentionally loose: they catch a broken tracker or a
# broken API call, not small quality differences between algorithms.
VALIDATE_MEAN_IOU = 0.45     # required mean IoU over the whole clip
VALIDATE_SUCCESS_IOU = 0.30  # per-frame IoU counted as "still on target"
VALIDATE_SUCCESS_RATE = 0.90 # required fraction of frames on target


def _resolve_creator(class_name):
    """Return the ``create`` callable for a tracker class, or None.

    OpenCV's Python bindings historically exposed two spellings:
    ``cv2.TrackerMIL_create()`` (flat, older) and ``cv2.TrackerMIL.create()``
    (class static method, current). Checking both keeps the script working
    across every supported 4.9+ and 5.x build.
    """
    # Prefer the flat function when present because it exists on both APIs.
    flat = getattr(cv2, f"{class_name}_create", None)
    if flat is not None:
        return flat
    # Fall back to the static method on the tracker class itself.
    klass = getattr(cv2, class_name, None)
    if klass is not None:
        return getattr(klass, "create", None)
    # Neither spelling exists: this build does not ship the tracker.
    return None


def _params_for(class_name):
    """Instantiate the ``<Tracker>_Params`` object for DNN trackers, or None."""
    # DNN trackers receive their model paths through a params struct that the
    # bindings expose as e.g. ``cv2.TrackerVit_Params``.
    params_class = getattr(cv2, f"{class_name}_Params", None)
    return params_class() if params_class is not None else None


def make_mil(_models_dir):
    """MIL (2009): boosting over bags of positive patches. Always in main."""
    creator = _resolve_creator("TrackerMIL")
    return creator() if creator else None


def make_kcf(_models_dir):
    """KCF (2014): kernelized correlation filter. Lives in opencv_contrib."""
    creator = _resolve_creator("TrackerKCF")
    return creator() if creator else None


def make_csrt(_models_dir):
    """CSRT (2017): channel and spatial reliability DCF. In opencv_contrib."""
    creator = _resolve_creator("TrackerCSRT")
    return creator() if creator else None


def make_dasiamrpn(models_dir):
    """DaSiamRPN (2018): siamese region-proposal tracker, three ONNX files."""
    creator = _resolve_creator("TrackerDaSiamRPN")
    params = _params_for("TrackerDaSiamRPN")
    if creator is None or params is None:
        return None
    # The params struct wants the backbone plus two correlation kernels.
    model = models_dir / "dasiamrpn_model.onnx"
    kernel_cls = models_dir / "dasiamrpn_kernel_cls1.onnx"
    kernel_r1 = models_dir / "dasiamrpn_kernel_r1.onnx"
    # Missing files mean the tracker is unavailable, not that we should crash;
    # download_models.py explains how to fetch them.
    if not (model.exists() and kernel_cls.exists() and kernel_r1.exists()):
        return None
    params.model = str(model)
    params.kernel_cls1 = str(kernel_cls)
    params.kernel_r1 = str(kernel_r1)
    return creator(params)


def make_nanotrack(models_dir):
    """NanoTrack v2 (2022): ~2 MB siamese tracker built for edge devices."""
    creator = _resolve_creator("TrackerNano")
    params = _params_for("TrackerNano")
    if creator is None or params is None:
        return None
    backbone = models_dir / "nanotrack_backbone_sim.onnx"
    head = models_dir / "nanotrack_head_sim.onnx"
    if not (backbone.exists() and head.exists()):
        return None
    # NanoTrack splits its network into a backbone and a neck+head pair.
    params.backbone = str(backbone)
    params.neckhead = str(head)
    return creator(params)


def make_vittrack(models_dir):
    """VitTrack (2023): transformer tracker, OpenCV 5's flagship replacement
    for the removed GOTURN."""
    creator = _resolve_creator("TrackerVit")
    params = _params_for("TrackerVit")
    if creator is None or params is None:
        return None
    net = models_dir / "object_tracking_vittrack_2023sep.onnx"
    if not net.exists():
        return None
    # A single ONNX file holds the whole model; it is under 1 MB.
    params.net = str(net)
    return creator(params)


# Registry mapping the CLI name to a builder. Order matters only for help
# text; availability is decided at runtime by each builder.
TRACKER_BUILDERS = {
    "mil": make_mil,
    "kcf": make_kcf,
    "csrt": make_csrt,
    "dasiamrpn": make_dasiamrpn,
    "nanotrack": make_nanotrack,
    "vittrack": make_vittrack,
}


def create_tracker(name, models_dir):
    """Build the requested tracker or raise a helpful error explaining why not."""
    builder = TRACKER_BUILDERS[name]
    tracker = builder(models_dir)
    if tracker is not None:
        return tracker
    # Distinguish "not in this OpenCV build" from "model files not downloaded"
    # because the user fixes the two problems differently.
    if name in ("kcf", "csrt"):
        raise RuntimeError(
            f"{name} requires an opencv_contrib build "
            "(pip install opencv-contrib-python)."
        )
    if name in ("dasiamrpn", "nanotrack", "vittrack"):
        raise RuntimeError(
            f"{name} model files not found in {models_dir}. "
            "Run: python3 download_models.py"
        )
    raise RuntimeError(f"{name} is not available in this OpenCV build.")


def list_trackers(models_dir):
    """Print one line per tracker with its availability in this environment."""
    print(f"OpenCV version: {cv2.__version__}")
    for name, builder in TRACKER_BUILDERS.items():
        # Building the tracker is the only reliable availability test.
        try:
            available = builder(models_dir) is not None
        except cv2.error:
            available = False
        print(f"  {name:<10} {'available' if available else 'NOT available'}")


def iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes; 0 when disjoint."""
    # Convert to corner coordinates for the overlap computation.
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3]
    # The intersection rectangle is bounded by the inner edges.
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    # Union = sum of areas minus the double-counted intersection.
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def make_synthetic_video(
        path, size=SYNTH_SIZE, frames=SYNTH_FRAMES, fps=30.0):
    """Write a deterministic clip of a textured square on a noisy background.

    Returns the list of ground-truth (x, y, w, h) boxes, one per frame. Both
    the validation mode and the regression tests rely on this function, and
    the C++ example generates the same scene from the same parameters so the
    two implementations can be compared numerically.
    """
    width, height = size
    # A fixed seed makes the background and target identical on every run.
    rng = np.random.RandomState(SYNTH_SEED)
    # Background: low-contrast noise over a horizontal gradient. Texture
    # everywhere prevents trackers from locking onto a blank frame edge.
    gradient = np.tile(np.linspace(60, 120, width, dtype=np.uint8), (height, 1))
    noise = rng.randint(0, 40, (height, width), dtype=np.uint8)
    background = cv2.merge([
        cv2.add(gradient, noise),
        cv2.add(gradient, rng.randint(0, 40, (height, width), dtype=np.uint8)),
        cv2.add(gradient, rng.randint(0, 40, (height, width), dtype=np.uint8)),
    ])
    # Target: an 8x8 grid of random bright colors gives strong, unique
    # texture that every tracker family can latch onto.
    target = rng.randint(64, 255, (8, 8, 3), dtype=np.uint8)
    target = cv2.resize(target, (SYNTH_TARGET, SYNTH_TARGET),
                        interpolation=cv2.INTER_NEAREST)
    # MJPG in an AVI container is widely available in desktop OpenCV builds.
    # The explicit isOpened() check below reports a clear error on builds that
    # were configured without a compatible writer backend.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {path}")
    boxes = []
    # Sweep the target along a smooth Lissajous-style path. The speed tops
    # out near 10 px/frame: fast enough to expose a broken tracker, slow
    # enough that even MIL's small search window can physically follow.
    for index in range(frames):
        phase = index / frames
        x = int((width - SYNTH_TARGET - 40) * 0.5 *
                (1 + np.sin(np.pi * phase - np.pi / 2)) + 20)
        y = int((height - SYNTH_TARGET - 40) * 0.5 *
                (1 + np.sin(2 * np.pi * phase)) + 20)
        # Compose the frame: copy the background, paste the target.
        frame = background.copy()
        frame[y:y + SYNTH_TARGET, x:x + SYNTH_TARGET] = target
        writer.write(frame)
        boxes.append((x, y, SYNTH_TARGET, SYNTH_TARGET))
    writer.release()
    return boxes


def open_capture(source):
    """Open a video file or a numeric camera index and fail with a clear error."""
    # A purely numeric argument selects a camera; anything else is a path.
    capture = (cv2.VideoCapture(int(source)) if str(source).isdigit()
               else cv2.VideoCapture(str(source)))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input: {source}")
    return capture


def parse_bbox(text):
    """Parse an exact ``x,y,w,h`` box and reject malformed dimensions."""
    # Splitting first prevents partial parses such as "10,20,30,40junk".
    fields = text.split(",")
    if len(fields) != 4:
        raise RuntimeError(
            f"Cannot parse --bbox={text!r}; expected exactly x,y,w,h"
        )
    try:
        box = tuple(int(field) for field in fields)
    except ValueError as error:
        raise RuntimeError(
            f"Cannot parse --bbox={text!r}; every value must be an integer"
        ) from error
    # A tracker cannot initialize from an empty or inverted rectangle.
    if box[2] <= 0 or box[3] <= 0:
        raise RuntimeError("--bbox width and height must both be positive")
    return box


def _validate_bbox_in_frame(box, frame):
    """Return an integer box after proving that it lies inside ``frame``."""
    # Convert selector-provided numeric values as well as CLI values uniformly.
    x, y, width, height = (int(value) for value in box)
    frame_height, frame_width = frame.shape[:2]
    if (x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > frame_width or y + height > frame_height):
        raise RuntimeError(
            "--bbox must have positive size and lie fully inside the first "
            f"frame ({frame_width}x{frame_height})"
        )
    return x, y, width, height


def track(tracker, capture, init_box, args, ground_truth=None):
    """Core loop shared by normal runs and validation.

    Reads frames, updates the tracker, draws and optionally displays or
    writes the annotated result, and returns a metrics dictionary.
    """
    # Read the first frame; the tracker is initialized on it.
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Input has no frames")
    # If the caller gave no box, ask the user to draw one (GUI builds only).
    if init_box is None:
        if args.no_display:
            raise RuntimeError("--bbox is required when --no-display is set")
        init_box = cv2.selectROI("Select object", frame, showCrosshair=True)
        cv2.destroyWindow("Select object")
    # Validate both CLI boxes and interactively selected boxes before asking
    # the tracker to consume them; this replaces opaque OpenCV assertions with
    # an actionable message.
    init_box = _validate_bbox_in_frame(init_box, frame)
    # init() learns the appearance model from the first frame and box.
    tracker.init(frame, init_box)
    # Prepare the optional annotated-video writer in the output directory.
    writer = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        # Create the directory explicitly instead of failing on write.
        output_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        # Preserve the source playback rate. Camera backends and a few unusual
        # containers report zero or NaN, in which case 30 FPS is a safe
        # documented fallback for the output container.
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        writer = cv2.VideoWriter(
            str(output_dir / f"tracked_{args.tracker}.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"), source_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write output video in {output_dir}")
    # TickMeter gives portable, monotonic per-frame timing for the FPS overlay.
    meter = cv2.TickMeter()
    ious = []            # per-frame IoU against ground truth, if provided
    frames_done = 1      # the init frame counts toward the total
    lost_frames = 0      # frames where update() reported failure
    if writer is not None:
        # Include the initialized frame so the annotated output has exactly
        # the same number of processed frames reported in the metrics. Draw on
        # a copy because the tracker has already learned from the clean frame.
        first_output = frame.copy()
        x, y, width, height = init_box
        cv2.rectangle(
            first_output, (x, y), (x + width, y + height), (0, 255, 0), 2
        )
        cv2.putText(
            first_output, f"{args.tracker}    0.0 FPS", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 170, 50), 2
        )
        writer.write(first_output)
    while True:
        # Stop early when a frame budget was requested (useful for tests).
        if args.max_frames and frames_done >= args.max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break  # end of stream
        # Time exactly the tracker update, not drawing or I/O.
        meter.start()
        found, box = tracker.update(frame)
        meter.stop()
        if found:
            # Draw the tracked box; integer coordinates for the raster calls.
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Be explicit on screen when the tracker declares failure.
            lost_frames += 1
            cv2.putText(frame, "tracking failure", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # Score against ground truth while validating on the synthetic clip.
        if ground_truth is not None and frames_done < len(ground_truth):
            truth = ground_truth[frames_done]
            ious.append(iou(box, truth) if found else 0.0)
        # Overlay the running average FPS of tracker updates.
        fps = meter.getFPS()
        cv2.putText(frame, f"{args.tracker}  {fps:5.1f} FPS", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 170, 50), 2)
        if writer is not None:
            writer.write(frame)
        # Count the frame before a possible interactive ESC exit because it
        # has already been read, tracked, scored, drawn, and written.
        frames_done += 1
        # Display unless the caller runs headless; ESC quits interactively.
        if not args.no_display:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    # Release resources deterministically; important on Windows file locks.
    capture.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    # Bundle everything a test or benchmark needs into one dictionary.
    return {
        "tracker": args.tracker,
        "opencv_version": cv2.__version__,
        "frames": frames_done,
        "lost_frames": lost_frames,
        "mean_fps": round(meter.getFPS(), 2),
        # Keep full precision for validation. Rounding before comparing a value
        # with its threshold can make Python disagree with the C++ example at
        # the boundary.
        "mean_iou": float(np.mean(ious)) if ious else None,
        "success_rate": (
            float(np.mean([v > VALIDATE_SUCCESS_IOU for v in ious]))
            if ious else None),
    }


def run(args):
    """Normal mode: track a user-supplied video or camera stream."""
    models_dir = Path(args.models_dir)
    tracker = create_tracker(args.tracker, models_dir)
    capture = open_capture(args.input)
    # Parse "x,y,w,h" once here so track() receives a ready-to-use tuple.
    init_box = parse_bbox(args.bbox) if args.bbox else None
    metrics = track(tracker, capture, init_box, args)
    _write_metrics(metrics, args)
    print(json.dumps(metrics, indent=2))
    return 0


def validate(args):
    """Validation mode: synthesize a clip with known ground truth and verify
    that the tracker follows the target within the documented thresholds."""
    models_dir = Path(args.models_dir)
    tracker = create_tracker(args.tracker, models_dir)
    # The synthetic clip lives in the output directory (or a default) so a
    # test can inspect it and nothing pollutes the source tree.
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)
    clip_path = output_dir / "synthetic_clip.avi"
    ground_truth = make_synthetic_video(clip_path)
    capture = open_capture(clip_path)
    # Initialize from the exact ground-truth box of frame zero.
    metrics = track(tracker, capture, ground_truth[0], args, ground_truth)
    _write_metrics(metrics, args)
    print(json.dumps(metrics, indent=2))
    # Apply the pass criteria and print an unambiguous marker for CI greps.
    # A short high-quality prefix is not a complete regression. Require all
    # generated frames in addition to the tracking-quality thresholds.
    passed = (metrics["frames"] == len(ground_truth)
              and metrics["mean_iou"] is not None
              and metrics["mean_iou"] >= VALIDATE_MEAN_IOU
              and metrics["success_rate"] >= VALIDATE_SUCCESS_RATE)
    print("VALIDATION PASSED" if passed else "VALIDATION FAILED")
    return 0 if passed else 1


def _write_metrics(metrics, args):
    """Persist metrics as JSON next to the annotated video when requested."""
    if args.output_dir is not None:
        metrics_path = Path(args.output_dir) / f"metrics_{args.tracker}.json"
        try:
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        except OSError as error:
            raise RuntimeError(
                f"Cannot write metrics file {metrics_path}: {error}"
            ) from error


def parse_args(argv=None):
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracker", choices=sorted(TRACKER_BUILDERS),
                        default="vittrack",
                        help="which tracker to run (default: vittrack)")
    parser.add_argument("--input", default="0",
                        help="video path or camera index (default: 0)")
    parser.add_argument("--bbox", default=None,
                        help="initial box as x,y,w,h; drawn interactively if omitted")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR),
                        help="directory holding the ONNX model files")
    parser.add_argument("--output-dir", default=None,
                        help="write annotated video and metrics JSON here")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after this many frames (0 = whole stream)")
    parser.add_argument("--no-display", action="store_true",
                        help="run headless: no windows, no waitKey")
    parser.add_argument("--validate", action="store_true",
                        help="run the synthetic-clip regression check")
    parser.add_argument("--list-trackers", action="store_true",
                        help="report tracker availability and exit")
    arguments = parser.parse_args(argv)
    if arguments.max_frames < 0:
        parser.error("--max-frames must be zero or a positive integer")
    return arguments


def main(argv=None):
    """Dispatch to list/validate/run and convert errors into clean exits."""
    args = parse_args(argv)
    if args.list_trackers:
        list_trackers(Path(args.models_dir))
        return 0
    try:
        return validate(args) if args.validate else run(args)
    except (RuntimeError, OSError, cv2.error) as error:
        # A readable one-line error beats a traceback for tutorial code.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

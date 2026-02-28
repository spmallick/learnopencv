"""
YuNet Webcam Face Anonymizer (Tutorial Version)

This script shows how to:
1) Load OpenCV's YuNet neural face detector (FaceDetectorYN).
2) Read frames from a webcam in real time.
3) Detect faces in each frame.
4) Apply one of two anonymization effects on each detected face:
   - Gaussian blur
   - Pixelation (mosaic)
5) Display the processed stream live.

Typical run command:
    python yunet_webcam_face_blur.py \
        --model face_detection_yunet_2023mar.onnx \
        --mode pixelate
"""

import argparse
from pathlib import Path

import cv2


# If the user does not pass --model, we try these filenames in the script folder.
# Both are official YuNet model filenames commonly seen in tutorials/repos.
DEFAULT_MODEL_CANDIDATES = [
    "face_detection_yunet_2023mar.onnx",
    "face_detection_yunet_2022mar.onnx",
]


def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Keeping parameters configurable from CLI makes the tutorial practical:
    users can tune detector confidence, camera index, and anonymization mode
    without editing source code.
    """
    parser = argparse.ArgumentParser(
        description="Real-time webcam face detection + blur/pixelate using OpenCV YuNet"
    )

    # Path to ONNX model. If omitted, we auto-search common names.
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YuNet ONNX model. If omitted, script tries common filenames in this folder.",
    )

    # Webcam index (0 is usually default webcam, 1/2... for external cameras).
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")

    # Minimum confidence for detected face boxes.
    # Increase to reduce false positives; decrease if detections are missed.
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.9,
        help="Minimum face confidence score (default: 0.9)",
    )

    # NMS threshold (Non-Maximum Suppression) merges overlapping detections.
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.3,
        help="NMS threshold for YuNet (default: 0.3)",
    )

    # Max number of raw detections before NMS.
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Maximum number of detections before NMS (default: 5000)",
    )

    # Optional visualization aid: draw confidence scores.
    parser.add_argument(
        "--show-score",
        action="store_true",
        help="Draw detection score on each face",
    )

    # Choose anonymization method.
    parser.add_argument(
        "--mode",
        type=str,
        choices=["blur", "pixelate"],
        default="blur",
        help="Face anonymization mode: blur or pixelate (default: blur)",
    )

    # Pixelation control: bigger block => stronger mosaic effect.
    parser.add_argument(
        "--pixel-block-size",
        type=int,
        default=16,
        help="Pixel block size for pixelation mode (default: 16)",
    )

    return parser.parse_args()


def resolve_model_path(cli_path: str | None) -> Path:
    """Resolve and validate the YuNet model path.

    Priority:
    1) If user passed --model, use it.
    2) Otherwise look for known filenames in the script directory.
    """
    if cli_path:
        model_path = Path(cli_path).expanduser().resolve()
        if model_path.is_file():
            return model_path
        raise FileNotFoundError(f"Model not found: {model_path}")

    script_dir = Path(__file__).resolve().parent
    for name in DEFAULT_MODEL_CANDIDATES:
        candidate = script_dir / name
        if candidate.is_file():
            return candidate

    expected = " or ".join(DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        "YuNet model not found. Pass --model <path-to-onnx> or place "
        f"{expected} in {script_dir}"
    )


def clamp_rect(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int):
    """Clip a rectangle to frame boundaries.

    Detection boxes can occasionally touch/overflow edges.
    Clamping prevents slicing errors when extracting face ROI.
    Returns None if resulting box is invalid (zero or negative area).
    """
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def blur_face_region(frame, rect):
    """Apply Gaussian blur to one detected face ROI.

    We scale kernel size with face size so blur intensity looks consistent
    across near/far faces.
    """
    x1, y1, x2, y2 = rect

    # Region of interest (face crop from the frame).
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    face_w = x2 - x1
    face_h = y2 - y1

    # Kernel size for Gaussian blur must be odd.
    # We derive it from face dimensions for adaptive strength.
    kernel = max(15, ((min(face_w, face_h) // 3) | 1))

    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    frame[y1:y2, x1:x2] = blurred


def pixelate_face_region(frame, rect, block_size: int):
    """Apply mosaic pixelation to one detected face ROI.

    Technique:
    1) Downscale ROI heavily.
    2) Upscale back with nearest-neighbor interpolation.
    This creates large square color blocks.
    """
    x1, y1, x2, y2 = rect
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    h, w = roi.shape[:2]

    # Guard against invalid block size and tiny ROIs.
    safe_block = max(1, block_size)

    # Small intermediate image dimensions.
    # Larger block size -> smaller intermediate image -> stronger pixelation.
    small_w = max(1, w // safe_block)
    small_h = max(1, h // safe_block)

    # First resize: compress details.
    temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)

    # Second resize: expand with nearest neighbor to preserve "blocks".
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    # Write processed ROI back into original frame.
    frame[y1:y2, x1:x2] = pixelated


def main() -> None:
    """Main application loop."""
    args = parse_args()
    model_path = resolve_model_path(args.model)

    # Open camera stream.
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    # Read one frame first so we know actual camera resolution.
    # YuNet detector needs input size configured explicitly.
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Cannot read initial frame from webcam")

    frame_h, frame_w = frame.shape[:2]

    # Create YuNet detector.
    # Signature: FaceDetectorYN.create(model, config, input_size, score_thr, nms_thr, top_k)
    detector = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (frame_w, frame_h),
        args.score_threshold,
        args.nms_threshold,
        args.top_k,
    )

    print(f"Using model: {model_path}")
    print(f"Mode: {args.mode}")
    print("Press 'q' or ESC to quit")

    while True:
        # Read next webcam frame.
        ok, frame = cap.read()
        if not ok or frame is None:
            # End loop gracefully if camera stops providing frames.
            break

        h, w = frame.shape[:2]

        # Important: keep detector input size in sync with frame size.
        # If resolution changes, detector must be updated.
        detector.setInputSize((w, h))

        # Run neural face detection.
        # faces is either None or an ndarray shaped (N, 15):
        # [x, y, w, h, 5 landmarks (10 values), score]
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                # Face bounding box is first 4 values.
                x, y, bw, bh = face[:4].astype(int)

                # Keep box inside frame to avoid slicing issues.
                rect = clamp_rect(x, y, bw, bh, w, h)
                if rect is None:
                    continue

                # Apply selected anonymization effect.
                if args.mode == "pixelate":
                    pixelate_face_region(frame, rect, args.pixel_block_size)
                else:
                    blur_face_region(frame, rect)

                # Optional score overlay for debugging/tuning thresholds.
                if args.show_score:
                    score = float(face[-1])
                    x1, y1, x2, y2 = rect
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 255, 40), 2)
                    cv2.putText(
                        frame,
                        f"{score:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (40, 255, 40),
                        2,
                        cv2.LINE_AA,
                    )

        # Show processed frame in a window.
        cv2.imshow("YuNet Face Blur/Pixelate", frame)

        # waitKey(1) processes UI events and captures keyboard input.
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    # Always release resources cleanly.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

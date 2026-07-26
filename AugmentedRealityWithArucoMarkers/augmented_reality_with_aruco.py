#!/usr/bin/env python3
"""Replace a four-marker quadrilateral with an image using current ArUco APIs."""

# Original example by Sunita Nayak at BigVision LLC, based on OpenCV.
# Modernized for current OpenCV APIs and reproducible headless execution.

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2 as cv
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = PROJECT_DIR / "test.jpg"
DEFAULT_OVERLAY = PROJECT_DIR / "new_scenery.jpg"
REQUIRED_MARKER_CORNERS = ((25, 1), (33, 2), (30, 0), (23, 0))


def create_detector():
    """Create the modern stateful ArUco detector once for all frames."""
    if not hasattr(cv, "aruco") or not hasattr(cv.aruco, "ArucoDetector"):
        raise RuntimeError(
            "This OpenCV build does not provide cv.aruco.ArucoDetector. "
            "Install OpenCV 4.8 or newer."
        )
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters()
    return cv.aruco.ArucoDetector(dictionary, parameters)


def destination_points(
    marker_corners,
    marker_ids,
    margin_fraction: float = 0.02,
) -> np.ndarray | None:
    """Return the four destination corners, or None if a marker is absent."""
    if marker_ids is None:
        return None
    if margin_fraction < 0:
        raise ValueError("margin_fraction must not be negative.")

    ids = np.asarray(marker_ids).reshape(-1)
    corners_by_id = {
        int(marker_id): np.asarray(marker_corners[index]).reshape(4, 2)
        for index, marker_id in enumerate(ids)
    }
    if any(marker_id not in corners_by_id for marker_id, _ in REQUIRED_MARKER_CORNERS):
        return None

    references = [
        corners_by_id[marker_id][corner_index]
        for marker_id, corner_index in REQUIRED_MARKER_CORNERS
    ]
    top_left, top_right, bottom_right, bottom_left = references
    top_edge_length = float(np.linalg.norm(top_left - top_right))
    if top_edge_length <= 1.0:
        raise ValueError("The top marker corners are too close to define a region.")
    margin = margin_fraction * top_edge_length

    return np.asarray(
        [
            [top_left[0] - margin, top_left[1] - margin],
            [top_right[0] + margin, top_right[1] - margin],
            [bottom_right[0] + margin, bottom_right[1] + margin],
            [bottom_left[0] - margin, bottom_left[1] + margin],
        ],
        dtype=np.float32,
    )


def augment_frame(
    frame: np.ndarray,
    overlay: np.ndarray,
    detector=None,
    margin_fraction: float = 0.02,
) -> tuple[np.ndarray, bool, list[int]]:
    """Warp overlay into the marker frame and preserve input when markers are missing."""
    if frame is None or frame.size == 0:
        raise ValueError("The input frame is empty.")
    if overlay is None or overlay.size == 0:
        raise ValueError("The overlay image is empty.")
    if detector is None:
        detector = create_detector()

    marker_corners, marker_ids, _ = detector.detectMarkers(frame)
    detected_ids = (
        [] if marker_ids is None else sorted(int(value) for value in marker_ids.reshape(-1))
    )
    points_destination = destination_points(
        marker_corners, marker_ids, margin_fraction
    )
    if points_destination is None:
        return frame.copy(), False, detected_ids

    height, width = overlay.shape[:2]
    points_source = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv.getPerspectiveTransform(points_source, points_destination)
    warped = cv.warpPerspective(
        overlay,
        transform,
        (frame.shape[1], frame.shape[0]),
        flags=cv.INTER_CUBIC,
    )

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    polygon = np.rint(points_destination).astype(np.int32)
    cv.fillConvexPoly(mask, polygon, 255, lineType=cv.LINE_AA)
    erosion_element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.erode(mask, erosion_element, iterations=3)

    augmented = frame.copy()
    augmented[mask > 0] = warped[mask > 0]
    return augmented, True, detected_ids


def compose_output(
    original: np.ndarray,
    augmented: np.ndarray,
    augmented_only: bool,
) -> np.ndarray:
    return augmented if augmented_only else cv.hconcat([original, augmented])


def default_image_output(input_path: Path) -> Path:
    suffix = input_path.suffix if input_path.suffix else ".jpg"
    return input_path.with_name(f"{input_path.stem}_ar_out_py{suffix}")


def default_video_output(input_path: Path | None) -> Path:
    if input_path is None:
        return PROJECT_DIR / "ar_out_py.avi"
    return input_path.with_name(f"{input_path.stem}_ar_out_py.avi")


def process_image(
    input_path: Path,
    overlay: np.ndarray,
    detector,
    output_path: Path,
    augmented_only: bool = False,
    strict: bool = False,
    display: bool = False,
) -> tuple[bool, list[int]]:
    frame = cv.imread(str(input_path), cv.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Input image not found or unreadable: {input_path}")

    augmented, did_augment, detected_ids = augment_frame(frame, overlay, detector)
    if strict and not did_augment:
        raise RuntimeError(
            "Required marker IDs 25, 33, 30, and 23 were not all detected. "
            f"Detected: {detected_ids or 'none'}."
        )
    output = compose_output(frame, augmented, augmented_only)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(output_path), output):
        raise OSError(f"OpenCV could not write: {output_path}")

    if display:
        cv.imshow("AR using ArUco markers", output)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return did_augment, detected_ids


def process_video(
    video_path: Path | None,
    camera: int | None,
    overlay: np.ndarray,
    detector,
    output_path: Path,
    augmented_only: bool = False,
    strict: bool = False,
    display: bool = False,
) -> tuple[int, int]:
    capture = cv.VideoCapture(camera if camera is not None else str(video_path))
    if not capture.isOpened():
        source = f"camera {camera}" if camera is not None else str(video_path)
        raise RuntimeError(f"Could not open {source}.")

    fps = capture.get(cv.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 28.0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_count = 0
    augmented_count = 0
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            augmented, did_augment, _ = augment_frame(frame, overlay, detector)
            output = compose_output(frame, augmented, augmented_only)
            if writer is None:
                writer = cv.VideoWriter(
                    str(output_path),
                    cv.VideoWriter_fourcc(*"MJPG"),
                    fps,
                    (output.shape[1], output.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video: {output_path}")
            writer.write(output)
            frame_count += 1
            augmented_count += int(did_augment)

            if display:
                cv.imshow("AR using ArUco markers", output)
                key = cv.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv.destroyAllWindows()

    if frame_count == 0:
        raise RuntimeError("The input produced no readable frames.")
    if strict and augmented_count == 0:
        raise RuntimeError(
            "None of the frames contained all required marker IDs "
            "25, 33, 30, and 23."
        )
    return frame_count, augmented_count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace a four-ArUco-marker region with an image."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Input image.")
    source.add_argument("--video", type=Path, help="Input video.")
    source.add_argument("--camera", type=int, help="Camera device index.")
    parser.add_argument(
        "--overlay",
        type=Path,
        default=DEFAULT_OVERLAY,
        help="Image to place inside the marker region.",
    )
    parser.add_argument("--output", type=Path, help="Output image or AVI path.")
    parser.add_argument(
        "--augmented-only",
        action="store_true",
        help="Write only the augmented view instead of a side-by-side comparison.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the required marker set cannot be augmented.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show results in a GUI window; press Q or Escape to stop video.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_image = args.image
    if input_image is None and args.video is None and args.camera is None:
        input_image = DEFAULT_IMAGE

    try:
        overlay = cv.imread(str(args.overlay), cv.IMREAD_COLOR)
        if overlay is None:
            raise FileNotFoundError(
                f"Overlay image not found or unreadable: {args.overlay}"
            )
        detector = create_detector()

        if input_image is not None:
            output_path = args.output or default_image_output(input_image)
            did_augment, detected_ids = process_image(
                input_image,
                overlay,
                detector,
                output_path,
                augmented_only=args.augmented_only,
                strict=args.strict,
                display=args.display,
            )
            status = "augmented" if did_augment else "unchanged (markers missing)"
            print(
                f"Wrote {output_path}: {status}; "
                f"detected IDs {detected_ids or 'none'}"
            )
        else:
            output_path = args.output or default_video_output(args.video)
            frame_count, augmented_count = process_video(
                args.video,
                args.camera,
                overlay,
                detector,
                output_path,
                augmented_only=args.augmented_only,
                strict=args.strict,
                display=args.display,
            )
            print(
                f"Wrote {output_path}: augmented {augmented_count}/"
                f"{frame_count} frames"
            )
    except (FileNotFoundError, OSError, RuntimeError, ValueError, cv.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

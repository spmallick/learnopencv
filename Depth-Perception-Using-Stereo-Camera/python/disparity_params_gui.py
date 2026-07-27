"""Tune StereoBM on a real stereo rig, or run a bounded headless smoke test."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2

from stereo_depth import (
    DEFAULT_CONFIG,
    DEFAULT_MAPS,
    StereoBMConfig,
    compute_disparity,
    create_matcher,
    disparity_visualization,
    load_config,
    load_rectification_maps,
    rectify_pair,
    save_config,
)


WINDOW = "StereoBM disparity"


def _nothing(_: int) -> None:
    pass


def _create_trackbars(config: StereoBMConfig) -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 600)
    controls = (
        ("numDisparities/16", config.num_disparities // 16, 32),
        ("blockSize", (config.block_size - 5) // 2, 125),
        ("preFilterType", config.pre_filter_type, 1),
        ("preFilterSize", (config.pre_filter_size - 5) // 2, 125),
        ("preFilterCap", config.pre_filter_cap, 63),
        ("textureThreshold", config.texture_threshold, 100),
        ("uniquenessRatio", config.uniqueness_ratio, 100),
        ("speckleRange", config.speckle_range, 100),
        ("speckleWindowSize", config.speckle_window_size, 200),
        ("disp12MaxDiff+1", config.disp12_max_diff + 1, 65),
        ("minDisparity", max(0, config.min_disparity), 64),
    )
    for name, value, maximum in controls:
        cv2.createTrackbar(name, WINDOW, int(value), maximum, _nothing)


def _config_from_trackbars(base: StereoBMConfig) -> StereoBMConfig:
    return replace(
        base,
        num_disparities=max(
            1, cv2.getTrackbarPos("numDisparities/16", WINDOW)
        )
        * 16,
        block_size=cv2.getTrackbarPos("blockSize", WINDOW) * 2 + 5,
        pre_filter_type=cv2.getTrackbarPos("preFilterType", WINDOW),
        pre_filter_size=cv2.getTrackbarPos("preFilterSize", WINDOW) * 2 + 5,
        pre_filter_cap=max(1, cv2.getTrackbarPos("preFilterCap", WINDOW)),
        texture_threshold=cv2.getTrackbarPos("textureThreshold", WINDOW),
        uniqueness_ratio=cv2.getTrackbarPos("uniquenessRatio", WINDOW),
        speckle_range=cv2.getTrackbarPos("speckleRange", WINDOW),
        speckle_window_size=cv2.getTrackbarPos("speckleWindowSize", WINDOW),
        disp12_max_diff=cv2.getTrackbarPos("disp12MaxDiff+1", WINDOW) - 1,
        min_disparity=cv2.getTrackbarPos("minDisparity", WINDOW),
    ).validated()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-camera", type=int, default=2)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--save-config",
        type=Path,
        help="Write the final configuration; the input file is never overwritten.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not create GUI controls; useful for a real-rig smoke test.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; zero means run until Escape.",
    )
    parser.add_argument("--output", type=Path, help="Save the final disparity view.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.left_camera == args.right_camera:
        raise ValueError("Left and right camera indices must be different.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative.")
    if args.no_display and args.max_frames == 0:
        raise ValueError("--no-display requires a positive --max-frames.")

    maps = load_rectification_maps(args.maps)
    active_config = load_config(args.config)
    matcher = create_matcher(active_config)

    left_camera = cv2.VideoCapture(args.left_camera)
    right_camera = cv2.VideoCapture(args.right_camera)
    if not left_camera.isOpened():
        raise RuntimeError(f"Could not open left camera {args.left_camera}.")
    if not right_camera.isOpened():
        left_camera.release()
        raise RuntimeError(f"Could not open right camera {args.right_camera}.")

    if not args.no_display:
        _create_trackbars(active_config)

    final_view = None
    frame_count = 0
    try:
        while True:
            if not left_camera.grab() or not right_camera.grab():
                raise RuntimeError("Could not grab synchronized stereo frames.")
            left_ok, left_color = left_camera.retrieve()
            right_ok, right_color = right_camera.retrieve()
            if not left_ok or not right_ok:
                raise RuntimeError("Could not retrieve synchronized stereo frames.")

            if not args.no_display:
                updated = _config_from_trackbars(active_config)
                if updated != active_config:
                    active_config = updated
                    matcher = create_matcher(active_config)

            left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
            left_rectified, right_rectified = rectify_pair(
                left_gray, right_gray, maps
            )
            disparity = compute_disparity(
                left_rectified, right_rectified, active_config, matcher=matcher
            )
            final_view = disparity_visualization(disparity)
            frame_count += 1

            if not args.no_display:
                cv2.imshow(WINDOW, final_view)
                if cv2.waitKey(1) == 27:
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        left_camera.release()
        right_camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if final_view is None:
        raise RuntimeError("No stereo frames were processed.")
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), final_view):
            raise OSError(f"Could not write disparity output: {destination}")
        print(f"Saved final disparity view: {destination}")
    if args.save_config:
        print(f"Saved configuration: {save_config(active_config, args.save_config)}")
    print(f"Processed stereo frames: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

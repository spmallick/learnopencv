"""Real-rig obstacle visualization built on the tested stereo-depth core."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from stereo_depth import (
    DEFAULT_CONFIG,
    DEFAULT_MAPS,
    compute_disparity,
    create_matcher,
    disparity_to_depth,
    disparity_visualization,
    find_largest_obstacle,
    load_config,
    load_rectification_maps,
    rectify_pair,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-camera", type=int, default=2)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--minimum-depth", type=float, default=10.0)
    parser.add_argument("--safe-distance", type=float, default=100.0)
    parser.add_argument("--minimum-area-fraction", type=float, default=0.01)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.left_camera == args.right_camera:
        raise ValueError("Left and right camera indices must be different.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative.")
    if args.no_display and args.max_frames == 0:
        raise ValueError("--no-display requires a positive --max-frames.")

    config = load_config(args.config)
    maps = load_rectification_maps(args.maps)
    matcher = create_matcher(config)
    left_camera = cv2.VideoCapture(args.left_camera)
    right_camera = cv2.VideoCapture(args.right_camera)
    if not left_camera.isOpened():
        raise RuntimeError(f"Could not open left camera {args.left_camera}.")
    if not right_camera.isOpened():
        left_camera.release()
        raise RuntimeError(f"Could not open right camera {args.right_camera}.")

    final_canvas = None
    frame_count = 0
    try:
        while True:
            if not left_camera.grab() or not right_camera.grab():
                raise RuntimeError("Could not grab synchronized stereo frames.")
            left_ok, left_color = left_camera.retrieve()
            right_ok, right_color = right_camera.retrieve()
            if not left_ok or not right_ok:
                raise RuntimeError("Could not retrieve synchronized stereo frames.")

            left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
            left_rectified, right_rectified = rectify_pair(
                left_gray, right_gray, maps
            )
            disparity = compute_disparity(
                left_rectified, right_rectified, config, matcher=matcher
            )
            depth = disparity_to_depth(disparity, config)
            _, obstacle = find_largest_obstacle(
                depth,
                min_depth=args.minimum_depth,
                max_depth=args.safe_distance,
                minimum_area_fraction=args.minimum_area_fraction,
            )

            final_canvas = cv2.remap(
                left_color,
                maps.left_x,
                maps.left_y,
                cv2.INTER_LANCZOS4,
                cv2.BORDER_CONSTANT,
            )
            if obstacle is None:
                cv2.putText(
                    final_canvas,
                    "SAFE",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
            else:
                x, y, width, height = obstacle.bounding_box
                cv2.rectangle(
                    final_canvas,
                    (x, y),
                    (x + width, y + height),
                    (0, 0, 255),
                    3,
                )
                cv2.putText(
                    final_canvas,
                    f"WARNING: {obstacle.mean_depth:.1f} cm",
                    (max(0, x), max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            frame_count += 1
            if not args.no_display:
                cv2.imshow("Obstacle avoidance", final_canvas)
                cv2.imshow("Disparity", disparity_visualization(disparity))
                if cv2.waitKey(1) == 27:
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        left_camera.release()
        right_camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if final_canvas is None:
        raise RuntimeError("No stereo frames were processed.")
    if args.output:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), final_canvas):
            raise OSError(f"Could not write output image: {destination}")
        print(f"Saved final obstacle view: {destination}")
    print(f"Processed stereo frames: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit disparity-to-depth scale and offset from a real stereo rig."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from stereo_depth import (
    DEFAULT_CONFIG,
    DEFAULT_MAPS,
    PROJECT_DIR,
    compute_disparity,
    create_matcher,
    disparity_visualization,
    fit_depth_model,
    load_config,
    load_rectification_maps,
    rectify_pair,
    save_config,
)


WINDOW = "Depth calibration disparity"
DEFAULT_OUTPUT_CONFIG = (
    PROJECT_DIR / "data" / "depth_estimation_params_py_updated.xml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-camera", type=int, default=2)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--max-distance", type=float, default=230.0)
    parser.add_argument("--min-distance", type=float, default=50.0)
    parser.add_argument("--sample-step", type=float, default=40.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.left_camera == args.right_camera:
        raise ValueError("Left and right camera indices must be different.")
    if (
        args.min_distance <= 0
        or args.max_distance <= args.min_distance
        or args.sample_step <= 0
    ):
        raise ValueError("Distance limits and sample step are invalid.")

    config = load_config(args.config)
    maps = load_rectification_maps(args.maps)
    matcher = create_matcher(config)
    state: dict[str, object] = {
        "disparity": None,
        "target_depth": float(args.max_distance),
        "disparities": [],
        "depths": [],
    }

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        disparity = state["disparity"]
        if not isinstance(disparity, np.ndarray):
            return
        if not 0 <= y < disparity.shape[0] or not 0 <= x < disparity.shape[1]:
            return
        value = float(disparity[y, x])
        if not np.isfinite(value) or value <= config.min_disparity:
            print("Ignored invalid disparity sample.")
            return
        target_depth = float(state["target_depth"])
        disparities = state["disparities"]
        depths = state["depths"]
        assert isinstance(disparities, list) and isinstance(depths, list)
        disparities.append(value)
        depths.append(target_depth)
        print(f"Depth {target_depth:.2f} cm -> disparity {value:.4f} px")
        state["target_depth"] = target_depth - float(args.sample_step)

    left_camera = cv2.VideoCapture(args.left_camera)
    right_camera = cv2.VideoCapture(args.right_camera)
    if not left_camera.isOpened():
        raise RuntimeError(f"Could not open left camera {args.left_camera}.")
    if not right_camera.isOpened():
        left_camera.release()
        raise RuntimeError(f"Could not open right camera {args.right_camera}.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 800, 600)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while float(state["target_depth"]) >= args.min_distance:
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
            state["disparity"] = disparity
            view = disparity_visualization(disparity)
            cv2.putText(
                view,
                f"Place target at {float(state['target_depth']):.0f} cm and click",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                255,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) == 27:
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()

    disparities = np.asarray(state["disparities"], dtype=np.float64)
    depths = np.asarray(state["depths"], dtype=np.float64)
    scale, offset, rmse = fit_depth_model(
        disparities, depths, min_disparity=config.min_disparity
    )
    calibrated = replace(
        config, depth_scale=scale, depth_offset=offset
    ).validated()
    destination = save_config(calibrated, args.output_config)
    print(f"Depth scale: {scale:.8f}")
    print(f"Depth offset: {offset:.8f}")
    print(f"Calibration RMSE: {rmse:.6f} cm")
    print(f"Saved calibrated configuration: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

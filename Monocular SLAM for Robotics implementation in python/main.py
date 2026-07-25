"""Educational monocular visual odometry and sparse mapping with OpenCV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from display import Display
from extractor import Frame, MatchResult, denormalize_point, match_frames
from pointmap import Map


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_DIR / "videos" / "car.mp4"


@dataclass(frozen=True)
class RunSummary:
    """Stable output metrics used by documentation and regression tests."""

    frames_processed: int
    pose_updates: int
    triangulated_points: int
    trajectory_path: Path
    frame_path: Path
    point_cloud_path: Path


def camera_matrix(width: int, height: int, focal_length: float) -> np.ndarray:
    """Construct a centered pinhole-camera intrinsic matrix."""

    return np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def triangulate(
    current_pose: np.ndarray,
    previous_pose: np.ndarray,
    current_points: np.ndarray,
    previous_points: np.ndarray,
) -> np.ndarray:
    """Triangulate normalized correspondences and keep points in front of both cameras."""

    homogeneous = cv2.triangulatePoints(
        current_pose[:3],
        previous_pose[:3],
        current_points.T,
        previous_points.T,
    ).T
    valid_scale = np.abs(homogeneous[:, 3]) > 1e-8
    homogeneous = homogeneous[valid_scale]
    homogeneous /= homogeneous[:, 3:4]

    current_depth = (current_pose @ homogeneous.T).T[:, 2]
    previous_depth = (previous_pose @ homogeneous.T).T[:, 2]
    finite = np.all(np.isfinite(homogeneous), axis=1)
    in_front = (current_depth > 0.0) & (previous_depth > 0.0)
    return homogeneous[finite & in_front, :3]


def draw_matches(
    image: np.ndarray,
    intrinsics: np.ndarray,
    current: Frame,
    previous: Frame,
    matches: MatchResult,
) -> np.ndarray:
    """Overlay inlier feature motion on a copy of the current frame."""

    annotated = image.copy()
    for current_point, previous_point in zip(
        current.points[matches.current_indices],
        previous.points[matches.previous_indices],
    ):
        current_pixel = denormalize_point(intrinsics, current_point)
        previous_pixel = denormalize_point(intrinsics, previous_point)
        cv2.line(
            annotated,
            current_pixel,
            previous_pixel,
            (255, 80, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            annotated,
            current_pixel,
            2,
            (0, 220, 255),
            -1,
            cv2.LINE_AA,
        )
    return annotated


def validate_map(
    map_state: Map,
    frames_processed: int,
    pose_updates: int,
) -> None:
    """Check stable geometric and numerical facts without overfitting exact poses."""

    if frames_processed < 2 or pose_updates < 1:
        raise RuntimeError("At least one two-view pose update is required")
    if len(map_state.points) < 8:
        raise RuntimeError(
            f"Expected at least eight triangulated points, got "
            f"{len(map_state.points)}"
        )
    for frame in map_state.frames:
        if frame.pose.shape != (4, 4) or not np.all(np.isfinite(frame.pose)):
            raise RuntimeError("Every camera pose must be a finite 4x4 matrix")
        if not np.allclose(frame.pose[3], [0.0, 0.0, 0.0, 1.0]):
            raise RuntimeError("Camera poses must remain homogeneous transforms")


def run(
    *,
    video_path: Path,
    output_dir: Path,
    max_frames: int,
    output_width: int,
    focal_length: float,
    show_windows: bool,
    validate: bool,
) -> RunSummary:
    """Run the visual-odometry front end and save its reproducible outputs."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open input video: {video_path}")

    map_state = Map()
    display = Display(960, 540) if show_windows else None
    frames_processed = 0
    pose_updates = 0
    last_annotated = None
    intrinsics = None

    while max_frames == 0 or frames_processed < max_frames:
        has_frame, source_frame = capture.read()
        if not has_frame or source_frame is None:
            break

        scale = output_width / source_frame.shape[1]
        output_height = max(1, round(source_frame.shape[0] * scale))
        image = cv2.resize(source_frame, (output_width, output_height))
        if intrinsics is None:
            intrinsics = camera_matrix(output_width, output_height, focal_length)

        frame = Frame(frames_processed, image, intrinsics)
        map_state.add_frame(frame)
        frames_processed += 1

        if len(map_state.frames) == 1:
            last_annotated = image.copy()
            continue

        previous = map_state.frames[-2]
        try:
            matches = match_frames(frame, previous)
        except RuntimeError as error:
            print(f"Frame {frame.frame_id}: skipped pose update ({error})")
            last_annotated = image.copy()
            continue

        frame.pose = matches.relative_pose @ previous.pose
        world_points = triangulate(
            frame.pose,
            previous.pose,
            frame.points[matches.current_indices],
            previous.points[matches.previous_indices],
        )
        map_state.add_points(world_points)
        pose_updates += 1
        last_annotated = draw_matches(
            image,
            intrinsics,
            frame,
            previous,
            matches,
        )

        if display is not None:
            key = display.show(last_annotated, delay_milliseconds=1)
            if key in (27, ord("q")):
                break

    capture.release()
    if display is not None:
        display.close()

    if last_annotated is None or frames_processed == 0:
        raise RuntimeError(f"Input video has no readable frames: {video_path}")
    if validate:
        validate_map(map_state, frames_processed, pose_updates)

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "slam-trajectory.png"
    frame_path = output_dir / "slam-feature-tracks.png"
    point_cloud_path = output_dir / "slam-map.ply"

    trajectory = map_state.render_top_down()
    if not cv2.imwrite(str(trajectory_path), trajectory):
        raise OSError(f"Unable to write trajectory image: {trajectory_path}")
    if not cv2.imwrite(str(frame_path), last_annotated):
        raise OSError(f"Unable to write feature-track image: {frame_path}")
    map_state.save_ply(point_cloud_path)

    if validate:
        for image_path in (trajectory_path, frame_path):
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                raise RuntimeError(f"Saved output is unreadable: {image_path}")
        if point_cloud_path.stat().st_size == 0:
            raise RuntimeError("Saved PLY point cloud is empty")

    return RunSummary(
        frames_processed,
        pose_updates,
        len(map_state.points),
        trajectory_path,
        frame_path,
        point_cloud_path,
    )


def parse_args() -> argparse.Namespace:
    """Parse video, camera, output, and automation options."""

    parser = argparse.ArgumentParser(
        description="Build a sparse map and monocular camera trajectory."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--focal-length", type=float, default=450.0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the tutorial application and provide clear command-line errors."""

    args = parse_args()
    try:
        if args.max_frames < 0:
            raise ValueError("max-frames cannot be negative")
        if args.width < 64:
            raise ValueError("width must be at least 64 pixels")
        if args.focal_length <= 0:
            raise ValueError("focal-length must be positive")

        summary = run(
            video_path=args.input.resolve(),
            output_dir=args.output_dir.resolve(),
            max_frames=args.max_frames,
            output_width=args.width,
            focal_length=args.focal_length,
            show_windows=not args.no_display,
            validate=args.validate,
        )
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Frames processed: {summary.frames_processed}")
        print(f"Pose updates: {summary.pose_updates}")
        print(f"Triangulated points: {summary.triangulated_points}")
        print(f"Trajectory image: {summary.trajectory_path}")
        print(f"Feature tracks: {summary.frame_path}")
        print(f"Point cloud: {summary.point_cloud_path}")
        if args.validate:
            print("VALIDATION PASSED: poses, map points, and outputs")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

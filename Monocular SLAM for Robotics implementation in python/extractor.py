"""Feature extraction and two-view pose estimation for monocular SLAM."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def add_ones(points: np.ndarray) -> np.ndarray:
    """Convert N×2 Euclidean points to N×3 homogeneous coordinates."""

    return np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1,
    )


def normalize_points(inverse_intrinsics: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map pixel coordinates into the normalized pinhole-camera plane."""

    normalized = (inverse_intrinsics @ add_ones(points).T).T
    return normalized[:, :2] / normalized[:, 2:3]


def denormalize_point(intrinsics: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    """Project one normalized point back to integer pixel coordinates."""

    projected = intrinsics @ np.array([point[0], point[1], 1.0])
    projected /= projected[2]
    return int(round(projected[0])), int(round(projected[1]))


def extract_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect Shi-Tomasi corners and compute ORB descriptors at those pixels."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=2500,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )
    if corners is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 32), dtype=np.uint8),
        )

    keypoints = [
        cv2.KeyPoint(float(point[0][0]), float(point[0][1]), 20)
        for point in corners
    ]
    keypoints, descriptors = cv2.ORB_create().compute(gray, keypoints)
    if not keypoints or descriptors is None:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 32), dtype=np.uint8),
        )

    points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    return points, descriptors


@dataclass
class Frame:
    """One image, its normalized features, descriptors, and world-to-camera pose."""

    frame_id: int
    image: np.ndarray
    intrinsics: np.ndarray
    pose: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    def __post_init__(self) -> None:
        pixel_points, self.descriptors = extract_features(self.image)
        self.points = normalize_points(
            np.linalg.inv(self.intrinsics),
            pixel_points,
        )


@dataclass(frozen=True)
class MatchResult:
    """Inlier feature indices and the previous-to-current camera transform."""

    current_indices: np.ndarray
    previous_indices: np.ndarray
    relative_pose: np.ndarray


def match_frames(current: Frame, previous: Frame) -> MatchResult:
    """Match ORB descriptors, estimate an essential matrix, and recover pose."""

    if len(current.descriptors) < 8 or len(previous.descriptors) < 8:
        raise RuntimeError("At least eight descriptors are required per frame")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidate_matches = matcher.knnMatch(
        current.descriptors,
        previous.descriptors,
        k=2,
    )

    current_indices: list[int] = []
    previous_indices: list[int] = []
    for neighbors in candidate_matches:
        if len(neighbors) != 2:
            continue
        best, second = neighbors
        if best.distance >= 0.75 * second.distance:
            continue

        current_point = current.points[best.queryIdx]
        previous_point = previous.points[best.trainIdx]
        if np.linalg.norm(current_point - previous_point) >= 0.15:
            continue
        current_indices.append(best.queryIdx)
        previous_indices.append(best.trainIdx)

    if len(current_indices) < 8:
        raise RuntimeError(
            f"Only {len(current_indices)} geometrically plausible matches found"
        )

    current_array = np.asarray(current_indices, dtype=np.int32)
    previous_array = np.asarray(previous_indices, dtype=np.int32)
    current_points = current.points[current_array]
    previous_points = previous.points[previous_array]

    essential, ransac_mask = cv2.findEssentialMat(
        current_points,
        previous_points,
        focal=1.0,
        pp=(0.0, 0.0),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.002,
    )
    if essential is None or ransac_mask is None:
        raise RuntimeError("Essential-matrix estimation failed")

    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        current_points,
        previous_points,
        focal=1.0,
        pp=(0.0, 0.0),
        mask=ransac_mask,
    )
    inliers = pose_mask.reshape(-1) != 0
    if int(np.count_nonzero(inliers)) < 8:
        raise RuntimeError("Pose recovery retained fewer than eight inliers")

    relative_pose = np.eye(4, dtype=np.float64)
    relative_pose[:3, :3] = rotation
    relative_pose[:3, 3] = translation.reshape(3)
    return MatchResult(
        current_array[inliers],
        previous_array[inliers],
        relative_pose,
    )

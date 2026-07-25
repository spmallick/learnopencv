"""Map storage plus headless trajectory and point-cloud output helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from extractor import Frame


@dataclass
class Map:
    """A compact teaching map containing frames and triangulated world points."""

    frames: list[Frame] = field(default_factory=list)
    points: list[np.ndarray] = field(default_factory=list)

    def add_frame(self, frame: Frame) -> None:
        """Append a frame after its pose and features have been initialized."""

        self.frames.append(frame)

    def add_points(self, world_points: np.ndarray) -> None:
        """Append finite 3D points while copying them away from temporary arrays."""

        for point in world_points:
            if np.all(np.isfinite(point)):
                self.points.append(np.asarray(point, dtype=np.float64).copy())

    def camera_centers(self) -> np.ndarray:
        """Return camera centers by inverting each world-to-camera pose."""

        centers = []
        for frame in self.frames:
            camera_to_world = np.linalg.inv(frame.pose)
            centers.append(camera_to_world[:3, 3])
        return np.asarray(centers, dtype=np.float64)

    def render_top_down(self, size: int = 800) -> np.ndarray:
        """Render X/Z camera motion and map points into a square BGR canvas."""

        if size < 100:
            raise ValueError("Trajectory image size must be at least 100")

        centers = self.camera_centers()
        points = (
            np.asarray(self.points, dtype=np.float64)
            if self.points
            else np.empty((0, 3), dtype=np.float64)
        )
        xz_sets = [centers[:, [0, 2]]]
        if len(points):
            xz_sets.append(points[:, [0, 2]])
        all_xz = np.concatenate(xz_sets, axis=0)

        lower = np.percentile(all_xz, 2, axis=0)
        upper = np.percentile(all_xz, 98, axis=0)
        span = np.maximum(upper - lower, 1e-6)
        margin = 40

        def project(values: np.ndarray) -> np.ndarray:
            normalized = (values - lower) / span
            pixels = margin + normalized * (size - 2 * margin)
            pixels[:, 1] = size - pixels[:, 1]
            return np.rint(pixels).astype(np.int32)

        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        if len(points):
            point_pixels = project(points[:, [0, 2]])
            for x, y in point_pixels[:: max(1, len(point_pixels) // 5000)]:
                cv2.circle(canvas, (int(x), int(y)), 1, (160, 160, 160), -1)

        center_pixels = project(centers[:, [0, 2]])
        if len(center_pixels) > 1:
            cv2.polylines(
                canvas,
                [center_pixels.reshape(-1, 1, 2)],
                False,
                (0, 140, 0),
                3,
                cv2.LINE_AA,
            )
        for x, y in center_pixels:
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 220), -1)

        cv2.putText(
            canvas,
            "Top-down monocular trajectory (X/Z)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def save_ply(self, path: Path) -> None:
        """Write triangulated points as a portable ASCII PLY point cloud."""

        path.parent.mkdir(parents=True, exist_ok=True)
        finite_points = [
            point for point in self.points if np.all(np.isfinite(point))
        ]
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(finite_points)}",
            "property float x",
            "property float y",
            "property float z",
            "end_header",
        ]
        lines.extend(
            f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}"
            for point in finite_points
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

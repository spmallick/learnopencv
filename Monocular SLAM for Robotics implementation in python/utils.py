"""Calibration-file helpers for KITTI-style monocular sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_calibration_file(path: Path) -> list[str]:
    """Read a small UTF-8 calibration file without assuming a working directory."""

    return path.read_text(encoding="utf-8").splitlines()


def extract_intrinsic_matrix(
    calibration_lines: list[str],
    camera_id: str = "P0",
) -> np.ndarray | None:
    """Extract the left 3×3 block of one KITTI 3×4 projection matrix."""

    for line in calibration_lines:
        if not line.startswith(f"{camera_id}:"):
            continue
        values = [float(value) for value in line.split()[1:]]
        if len(values) != 12:
            raise ValueError(
                f"{camera_id} must contain 12 projection values, got "
                f"{len(values)}"
            )
        projection = np.asarray(values, dtype=np.float64).reshape(3, 4)
        return projection[:, :3]
    return None

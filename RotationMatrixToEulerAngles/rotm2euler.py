#!/usr/bin/env python3
"""Convert between XYZ Tait-Bryan angles and 3x3 rotation matrices.

The convention used throughout this example is an active, right-handed
rotation of column vectors:

    R = Rz(z) @ Ry(y) @ Rx(x)

The returned angle vector is ordered ``[x, y, z]`` and is expressed in radians.
Euler/Tait-Bryan angles are not unique, so round-trip validation compares
rotation matrices rather than comparing angle vectors directly.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _as_finite_matrix(matrix: npt.ArrayLike) -> FloatArray:
    """Return *matrix* as a finite 3x3 float64 array or raise ValueError."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape (3, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("rotation matrix must contain only finite values")
    return array


def is_rotation_matrix(matrix: npt.ArrayLike, tolerance: float = 1e-9) -> bool:
    """Return True when *matrix* is a proper 3D rotation matrix.

    A proper rotation must be orthonormal and have determinant +1. Checking the
    determinant rejects reflections, which satisfy ``R.T @ R == I`` but are not
    rotations.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    try:
        rotation = _as_finite_matrix(matrix)
    except (TypeError, ValueError):
        return False

    identity_error = np.linalg.norm(
        rotation.T @ rotation - np.eye(3, dtype=np.float64), ord=np.inf
    )
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    return identity_error <= tolerance and determinant_error <= tolerance


def euler_angles_to_rotation_matrix(theta: npt.ArrayLike) -> FloatArray:
    """Build ``Rz(z) @ Ry(y) @ Rx(x)`` from ``[x, y, z]`` radians."""

    angles = np.asarray(theta, dtype=np.float64)
    if angles.shape != (3,):
        raise ValueError(f"theta must have shape (3,), got {angles.shape}")
    if not np.all(np.isfinite(angles)):
        raise ValueError("theta must contain only finite values")

    x_angle, y_angle, z_angle = (float(value) for value in angles)
    sin_x, cos_x = math.sin(x_angle), math.cos(x_angle)
    sin_y, cos_y = math.sin(y_angle), math.cos(y_angle)
    sin_z, cos_z = math.sin(z_angle), math.cos(z_angle)

    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


def rotation_matrix_to_euler_angles(
    matrix: npt.ArrayLike,
    *,
    validation_tolerance: float = 1e-9,
    singular_epsilon: float = 1e-9,
) -> FloatArray:
    """Return one valid ``[x, y, z]`` representation for a rotation matrix.

    At gimbal lock, infinitely many angle triples represent the same rotation.
    This implementation chooses ``z = 0`` and solves for ``x`` and ``y``.
    """

    rotation = _as_finite_matrix(matrix)
    if not is_rotation_matrix(rotation, validation_tolerance):
        raise ValueError("input is not a proper rotation matrix")
    if singular_epsilon <= 0:
        raise ValueError("singular_epsilon must be positive")

    horizontal_norm = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    singular = horizontal_norm < singular_epsilon

    if not singular:
        x_angle = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y_angle = math.atan2(-float(rotation[2, 0]), horizontal_norm)
        z_angle = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        x_angle = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        y_angle = math.atan2(-float(rotation[2, 0]), horizontal_norm)
        z_angle = 0.0

    return np.array([x_angle, y_angle, z_angle], dtype=np.float64)


# Compatibility aliases keep existing tutorial imports working.
isRotationMatrix = is_rotation_matrix
eulerAnglesToRotationMatrix = euler_angles_to_rotation_matrix
rotationMatrixToEulerAngles = rotation_matrix_to_euler_angles


def _matrix_round_trip_error(theta: npt.ArrayLike) -> float:
    """Return the infinity-norm error after an angles/matrix round trip."""

    original = euler_angles_to_rotation_matrix(theta)
    recovered_angles = rotation_matrix_to_euler_angles(original)
    recovered = euler_angles_to_rotation_matrix(recovered_angles)
    return float(np.linalg.norm(original - recovered, ord=np.inf))


def run_validation() -> dict[str, float | int]:
    """Run deterministic regression checks used by tests and CTest parity."""

    cases = [
        np.zeros(3, dtype=np.float64),
        np.array([math.pi / 2.0, 0.0, 0.0]),
        np.array([0.0, math.pi / 2.0, 0.0]),
        np.array([0.0, -math.pi / 2.0, 0.0]),
        np.array([0.3, -0.7, 1.2]),
        np.array([-2.4, math.pi / 2.0 - 1e-10, 0.8]),
    ]

    random_generator = np.random.default_rng(20260726)
    random_cases = random_generator.uniform(-math.pi, math.pi, size=(512, 3))
    errors = [_matrix_round_trip_error(case) for case in cases]
    errors.extend(_matrix_round_trip_error(case) for case in random_cases)
    maximum_error = max(errors)

    reflection = np.diag([1.0, 1.0, -1.0])
    if is_rotation_matrix(reflection):
        raise AssertionError("reflection matrix was incorrectly accepted")
    if maximum_error > 1e-9:
        raise AssertionError(f"round-trip error {maximum_error:.3e} exceeded tolerance")

    result: dict[str, float | int] = {
        "cases": len(errors),
        "max_matrix_error": maximum_error,
    }
    print(
        "VALIDATION PASSED: "
        f"{result['cases']} cases, max_matrix_error={maximum_error:.3e}"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Convert XYZ Tait-Bryan angles and 3x3 rotation matrices."
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--angles",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="angles ordered X Y Z; radians unless --degrees is present",
    )
    inputs.add_argument(
        "--matrix",
        nargs=9,
        type=float,
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="row-major 3x3 rotation matrix",
    )
    inputs.add_argument(
        "--validate",
        action="store_true",
        help="run deterministic regression checks",
    )
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="read and print angles in degrees instead of radians",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic command-line demonstration."""

    arguments = _build_parser().parse_args(argv)
    if arguments.validate:
        run_validation()
        return 0

    if arguments.matrix is not None:
        rotation = np.asarray(arguments.matrix, dtype=np.float64).reshape(3, 3)
        angles = rotation_matrix_to_euler_angles(rotation)
    else:
        supplied = arguments.angles or [20.0, -35.0, 70.0]
        angles = np.asarray(supplied, dtype=np.float64)
        if arguments.angles is None or arguments.degrees:
            angles = np.deg2rad(angles)
        rotation = euler_angles_to_rotation_matrix(angles)

    displayed_angles = np.rad2deg(angles) if arguments.degrees else angles
    units = "degrees" if arguments.degrees else "radians"
    print(f"Euler angles [x, y, z] ({units}):")
    print(np.array2string(displayed_angles, precision=10, suppress_small=True))
    print("Rotation matrix Rz @ Ry @ Rx:")
    print(np.array2string(rotation, precision=10, suppress_small=True))

    recovered = rotation_matrix_to_euler_angles(rotation)
    recovered_rotation = euler_angles_to_rotation_matrix(recovered)
    error = float(np.linalg.norm(rotation - recovered_rotation, ord=np.inf))
    print(f"Round-trip matrix infinity-norm error: {error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

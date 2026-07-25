#!/usr/bin/env python3
"""Create an average face from aligned portraits and 68-point landmarks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def read_dataset(directory: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read sorted JPEG images and their adjacent ``.jpg.txt`` landmark files."""
    images: list[np.ndarray] = []
    point_sets: list[np.ndarray] = []
    for image_path in sorted(directory.glob("*.jpg")):
        points_path = Path(f"{image_path}.txt")
        if not points_path.is_file():
            raise FileNotFoundError(f"Missing landmarks for {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        points = np.loadtxt(points_path, dtype=np.float32)
        if points.shape != (68, 2):
            raise ValueError(f"Expected 68 landmarks in {points_path}, got {points.shape}")
        images.append(image.astype(np.float32) / 255.0)
        point_sets.append(points)
    if len(images) < 2:
        raise ValueError(f"Need at least two image/landmark pairs in {directory}")
    return images, point_sets


def similarity_transform(
    input_points: np.ndarray, output_points: np.ndarray
) -> np.ndarray:
    """Estimate a similarity transform from two point pairs via a synthetic third."""
    sin60 = math.sin(math.radians(60.0))
    cos60 = math.cos(math.radians(60.0))
    source = np.asarray(input_points, dtype=np.float32).tolist()
    target = np.asarray(output_points, dtype=np.float32).tolist()

    for points in (source, target):
        x0, y0 = points[0]
        x1, y1 = points[1]
        points.append(
            [
                cos60 * (x0 - x1) - sin60 * (y0 - y1) + x1,
                sin60 * (x0 - x1) + cos60 * (y0 - y1) + y1,
            ]
        )

    transform, _ = cv2.estimateAffinePartial2D(
        np.asarray(source, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        method=cv2.LMEDS,
    )
    if transform is None:
        raise RuntimeError("Could not estimate a similarity transform")
    return transform


def rect_contains(rect: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x, y, width, height = rect
    return x <= point[0] < x + width and y <= point[1] < y + height


def calculate_delaunay_triangles(
    rect: tuple[int, int, int, int], points: np.ndarray
) -> list[tuple[int, int, int]]:
    subdiv = cv2.Subdiv2D(rect)
    for point in points:
        subdiv.insert((float(point[0]), float(point[1])))

    triangles: list[tuple[int, int, int]] = []
    for triangle in subdiv.getTriangleList():
        vertices = [
            (float(triangle[0]), float(triangle[1])),
            (float(triangle[2]), float(triangle[3])),
            (float(triangle[4]), float(triangle[5])),
        ]
        if not all(rect_contains(rect, point) for point in vertices):
            continue
        indices: list[int] = []
        for vertex in vertices:
            distances = np.linalg.norm(points - np.asarray(vertex), axis=1)
            index = int(np.argmin(distances))
            if distances[index] >= 1.0:
                break
            indices.append(index)
        if len(indices) == 3 and len(set(indices)) == 3:
            triangles.append(tuple(indices))
    if not triangles:
        raise RuntimeError("Delaunay triangulation produced no valid triangles")
    return triangles


def constrain_point(point: np.ndarray, width: int, height: int) -> tuple[float, float]:
    return (
        float(np.clip(point[0], 0, width - 1)),
        float(np.clip(point[1], 0, height - 1)),
    )


def warp_triangle(
    source: np.ndarray,
    destination: np.ndarray,
    source_triangle: list[tuple[float, float]],
    target_triangle: list[tuple[float, float]],
) -> None:
    source_rect = cv2.boundingRect(np.float32([source_triangle]))
    target_rect = cv2.boundingRect(np.float32([target_triangle]))
    sx, sy, sw, sh = source_rect
    tx, ty, tw, th = target_rect

    source_local = [(x - sx, y - sy) for x, y in source_triangle]
    target_local = [(x - tx, y - ty) for x, y in target_triangle]
    target_local_int = np.rint(target_local).astype(np.int32)

    mask = np.zeros((th, tw, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, target_local_int, (1.0, 1.0, 1.0), cv2.LINE_AA)

    source_patch = source[sy : sy + sh, sx : sx + sw]
    transform = cv2.getAffineTransform(
        np.float32(source_local), np.float32(target_local)
    )
    warped = cv2.warpAffine(
        source_patch,
        transform,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped *= mask
    target_view = destination[ty : ty + th, tx : tx + tw]
    target_view *= 1.0 - mask
    target_view += warped


def create_average_face(
    images: list[np.ndarray],
    point_sets: list[np.ndarray],
    width: int = 600,
    height: int = 600,
) -> tuple[np.ndarray, int]:
    if len(images) != len(point_sets):
        raise ValueError("Image and landmark counts must match")

    eye_target = np.float32([(0.3 * width, height / 3.0), (0.7 * width, height / 3.0)])
    boundary = np.float32(
        [
            (0, 0),
            (width / 2, 0),
            (width - 1, 0),
            (width - 1, height / 2),
            (width - 1, height - 1),
            (width / 2, height - 1),
            (0, height - 1),
            (0, height / 2),
        ]
    )

    normalized_images: list[np.ndarray] = []
    normalized_points: list[np.ndarray] = []
    average_points = np.zeros((68 + len(boundary), 2), dtype=np.float32)

    for image, points in zip(images, point_sets):
        transform = similarity_transform(points[[36, 45]], eye_target)
        normalized = cv2.warpAffine(image, transform, (width, height))
        transformed = cv2.transform(points.reshape(-1, 1, 2), transform).reshape(-1, 2)
        transformed = np.vstack((transformed, boundary)).astype(np.float32)
        average_points += transformed / len(images)
        normalized_images.append(normalized)
        normalized_points.append(transformed)

    triangles = calculate_delaunay_triangles((0, 0, width, height), average_points)
    output = np.zeros((height, width, 3), dtype=np.float32)
    for image, points in zip(normalized_images, normalized_points):
        warped = np.zeros_like(output)
        for triangle in triangles:
            source_triangle = [
                constrain_point(points[index], width, height) for index in triangle
            ]
            target_triangle = [
                constrain_point(average_points[index], width, height)
                for index in triangle
            ]
            warp_triangle(image, warped, source_triangle, target_triangle)
        output += warped
    return np.clip(output / len(images), 0.0, 1.0), len(triangles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "presidents")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "output" / "average-face.jpg")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width < 64 or args.height < 64:
        raise ValueError("Output dimensions must be at least 64 pixels")
    images, points = read_dataset(args.input_dir)
    output, triangle_count = create_average_face(
        images, points, args.width, args.height
    )
    output_8bit = np.rint(output * 255.0).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output_8bit):
        raise RuntimeError(f"Could not write output: {args.output}")

    print(f"OpenCV: {cv2.__version__}")
    print(f"Inputs: {len(images)}")
    print(f"Delaunay triangles: {triangle_count}")
    print(f"Saved: {args.output}")

    if args.validate:
        saved = cv2.imread(str(args.output))
        if (
            saved is None
            or saved.shape[:2] != (args.height, args.width)
            or triangle_count < 50
            or float(saved.std()) < 10.0
        ):
            raise RuntimeError("Average-face validation failed")
        print("Validation: PASS")

    if not args.no_display:
        cv2.imshow("Average face", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

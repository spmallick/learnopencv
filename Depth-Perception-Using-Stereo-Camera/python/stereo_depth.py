"""Reusable stereo-depth functions plus a headless image-pair command line."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAPS = PROJECT_DIR / "data" / "stereo_rectify_maps.xml"
DEFAULT_CONFIG = PROJECT_DIR / "data" / "depth_estmation_params_py.xml"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs"


@dataclass(frozen=True)
class StereoBMConfig:
    num_disparities: int = 144
    block_size: int = 35
    pre_filter_type: int = 1
    pre_filter_size: int = 9
    pre_filter_cap: int = 30
    texture_threshold: int = 10
    uniqueness_ratio: int = 6
    speckle_range: int = 29
    speckle_window_size: int = 50
    disp12_max_diff: int = 5
    min_disparity: int = 5
    depth_scale: float = 5257.782887811477
    depth_offset: float = 0.0

    def validated(self) -> "StereoBMConfig":
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16.")
        if not 5 <= self.block_size <= 255 or self.block_size % 2 == 0:
            raise ValueError("block_size must be odd and in [5, 255].")
        if self.pre_filter_type not in (0, 1):
            raise ValueError("pre_filter_type must be 0 or 1.")
        if not 5 <= self.pre_filter_size <= 255 or self.pre_filter_size % 2 == 0:
            raise ValueError("pre_filter_size must be odd and in [5, 255].")
        if not 1 <= self.pre_filter_cap <= 63:
            raise ValueError("pre_filter_cap must be in [1, 63].")
        for name in (
            "texture_threshold",
            "uniqueness_ratio",
            "speckle_range",
            "speckle_window_size",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.disp12_max_diff < -1:
            raise ValueError("disp12_max_diff must be -1 or greater.")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("depth_scale must be finite and positive.")
        if not np.isfinite(self.depth_offset):
            raise ValueError("depth_offset must be finite.")
        return self


@dataclass(frozen=True)
class RectificationMaps:
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray

    def validated(self) -> "RectificationMaps":
        maps = (self.left_x, self.left_y, self.right_x, self.right_y)
        if any(item is None or item.size == 0 for item in maps):
            raise ValueError("One or more rectification maps are empty.")
        if self.left_x.shape[:2] != self.left_y.shape[:2]:
            raise ValueError("Left rectification map shapes do not match.")
        if self.right_x.shape[:2] != self.right_y.shape[:2]:
            raise ValueError("Right rectification map shapes do not match.")
        if self.left_x.shape[:2] != self.right_x.shape[:2]:
            raise ValueError("Left and right rectification sizes do not match.")
        valid_pairs = (
            (
                self.left_x.dtype == np.int16
                and self.left_x.ndim == 3
                and self.left_x.shape[2] == 2
                and self.left_y.dtype == np.uint16
                and self.left_y.ndim == 2
            )
            or (
                self.left_x.dtype == np.float32
                and self.left_y.dtype == np.float32
                and self.left_x.ndim == self.left_y.ndim == 2
            ),
            (
                self.right_x.dtype == np.int16
                and self.right_x.ndim == 3
                and self.right_x.shape[2] == 2
                and self.right_y.dtype == np.uint16
                and self.right_y.ndim == 2
            )
            or (
                self.right_x.dtype == np.float32
                and self.right_y.dtype == np.float32
                and self.right_x.ndim == self.right_y.ndim == 2
            ),
        )
        if not all(valid_pairs):
            raise ValueError(
                "Rectification maps must use OpenCV fixed-point or "
                "single-channel float32 map pairs."
            )
        return self

    @property
    def image_size(self) -> tuple[int, int]:
        rows, columns = self.left_x.shape[:2]
        return columns, rows


@dataclass(frozen=True)
class Obstacle:
    bounding_box: tuple[int, int, int, int]
    mean_depth: float
    area_fraction: float


def _read_required_node(storage: cv2.FileStorage, name: str) -> float:
    node = storage.getNode(name)
    if node.empty():
        raise ValueError(f"Missing required configuration field: {name}")
    value = float(node.real())
    if not np.isfinite(value):
        raise ValueError(f"Configuration field {name} is not finite.")
    return value


def _read_optional_node(
    storage: cv2.FileStorage, name: str, default: float
) -> float:
    node = storage.getNode(name)
    return default if node.empty() else float(node.real())


def load_config(path: str | Path = DEFAULT_CONFIG) -> StereoBMConfig:
    config_path = Path(path).expanduser().resolve()
    storage = cv2.FileStorage(str(config_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open stereo configuration: {config_path}")
    try:
        values = {
            "num_disparities": int(_read_required_node(storage, "numDisparities")),
            "block_size": int(_read_required_node(storage, "blockSize")),
            "pre_filter_type": int(_read_required_node(storage, "preFilterType")),
            "pre_filter_size": int(_read_required_node(storage, "preFilterSize")),
            "pre_filter_cap": int(_read_required_node(storage, "preFilterCap")),
            "texture_threshold": int(
                _read_required_node(storage, "textureThreshold")
            ),
            "uniqueness_ratio": int(
                _read_required_node(storage, "uniquenessRatio")
            ),
            "speckle_range": int(_read_required_node(storage, "speckleRange")),
            "speckle_window_size": int(
                _read_required_node(storage, "speckleWindowSize")
            ),
            "disp12_max_diff": int(_read_required_node(storage, "disp12MaxDiff")),
            "min_disparity": int(_read_required_node(storage, "minDisparity")),
        }

        depth_scale_node = storage.getNode("depthScale")
        if depth_scale_node.empty():
            # Legacy files fitted depth against normalized disparity:
            # normalized = (disparity_px - minDisparity) / numDisparities.
            legacy_scale = _read_required_node(storage, "M")
            depth_scale = legacy_scale * values["num_disparities"]
        else:
            depth_scale = float(depth_scale_node.real())
        depth_offset = _read_optional_node(
            storage, "depthOffset", _read_optional_node(storage, "C", 0.0)
        )
    finally:
        storage.release()

    return StereoBMConfig(
        **values, depth_scale=depth_scale, depth_offset=depth_offset
    ).validated()


def save_config(config: StereoBMConfig, path: str | Path) -> Path:
    config.validated()
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(destination), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Could not open configuration for writing: {destination}")
    try:
        storage.write("numDisparities", config.num_disparities)
        storage.write("blockSize", config.block_size)
        storage.write("preFilterType", config.pre_filter_type)
        storage.write("preFilterSize", config.pre_filter_size)
        storage.write("preFilterCap", config.pre_filter_cap)
        storage.write("textureThreshold", config.texture_threshold)
        storage.write("uniquenessRatio", config.uniqueness_ratio)
        storage.write("speckleRange", config.speckle_range)
        storage.write("speckleWindowSize", config.speckle_window_size)
        storage.write("disp12MaxDiff", config.disp12_max_diff)
        storage.write("minDisparity", config.min_disparity)
        storage.write("depthScale", float(config.depth_scale))
        storage.write("depthOffset", float(config.depth_offset))
    finally:
        storage.release()
    return destination


def load_rectification_maps(
    path: str | Path = DEFAULT_MAPS,
) -> RectificationMaps:
    maps_path = Path(path).expanduser().resolve()
    storage = cv2.FileStorage(str(maps_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open rectification maps: {maps_path}")
    try:
        maps = RectificationMaps(
            storage.getNode("Left_Stereo_Map_x").mat(),
            storage.getNode("Left_Stereo_Map_y").mat(),
            storage.getNode("Right_Stereo_Map_x").mat(),
            storage.getNode("Right_Stereo_Map_y").mat(),
        )
    finally:
        storage.release()
    return maps.validated()


def create_matcher(config: StereoBMConfig) -> cv2.StereoBM:
    config.validated()
    matcher = cv2.StereoBM_create(
        numDisparities=config.num_disparities, blockSize=config.block_size
    )
    matcher.setPreFilterType(config.pre_filter_type)
    matcher.setPreFilterSize(config.pre_filter_size)
    matcher.setPreFilterCap(config.pre_filter_cap)
    matcher.setTextureThreshold(config.texture_threshold)
    matcher.setUniquenessRatio(config.uniqueness_ratio)
    matcher.setSpeckleRange(config.speckle_range)
    matcher.setSpeckleWindowSize(config.speckle_window_size)
    matcher.setDisp12MaxDiff(config.disp12_max_diff)
    matcher.setMinDisparity(config.min_disparity)
    return matcher


def _validate_pair(left: np.ndarray, right: np.ndarray) -> None:
    if left is None or right is None or left.size == 0 or right.size == 0:
        raise ValueError("Left and right images must be non-empty.")
    if left.shape != right.shape:
        raise ValueError("Left and right images must have the same shape.")
    if left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Stereo matching expects uint8 grayscale images.")


def rectify_pair(
    left: np.ndarray, right: np.ndarray, maps: RectificationMaps
) -> tuple[np.ndarray, np.ndarray]:
    _validate_pair(left, right)
    maps.validated()
    if (left.shape[1], left.shape[0]) != maps.image_size:
        raise ValueError(
            f"Image size {(left.shape[1], left.shape[0])} does not match "
            f"rectification map size {maps.image_size}."
        )
    left_rectified = cv2.remap(
        left, maps.left_x, maps.left_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT
    )
    right_rectified = cv2.remap(
        right, maps.right_x, maps.right_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT
    )
    return left_rectified, right_rectified


def compute_disparity(
    left: np.ndarray,
    right: np.ndarray,
    config: StereoBMConfig,
    *,
    matcher: cv2.StereoBM | None = None,
) -> np.ndarray:
    """Return disparity in pixels, using NaN for invalid/non-positive values."""

    _validate_pair(left, right)
    config.validated()
    active_matcher = matcher if matcher is not None else create_matcher(config)
    raw = active_matcher.compute(left, right).astype(np.float32) / 16.0
    raw[~np.isfinite(raw) | (raw <= config.min_disparity)] = np.nan
    return raw


def disparity_to_depth(
    disparity_pixels: np.ndarray, config: StereoBMConfig
) -> np.ndarray:
    """Convert pixel disparity with depth = scale/(d-min_d) + offset."""

    config.validated()
    disparity = np.asarray(disparity_pixels, dtype=np.float32)
    denominator = disparity - float(config.min_disparity)
    valid = np.isfinite(denominator) & (denominator > 0)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid] = (
        float(config.depth_scale) / denominator[valid]
        + float(config.depth_offset)
    )
    depth[~np.isfinite(depth)] = np.nan
    return depth


def fit_depth_model(
    disparity_pixels: np.ndarray,
    measured_depths: np.ndarray,
    *,
    min_disparity: float = 0.0,
) -> tuple[float, float, float]:
    """Fit depth = scale/(disparity-min_disparity) + offset."""

    disparity = np.asarray(disparity_pixels, dtype=np.float64).reshape(-1)
    depth = np.asarray(measured_depths, dtype=np.float64).reshape(-1)
    if disparity.shape != depth.shape:
        raise ValueError("Disparity and depth samples must have matching shapes.")
    denominator = disparity - float(min_disparity)
    valid = np.isfinite(denominator) & np.isfinite(depth) & (denominator > 0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("At least two finite positive-disparity samples are required.")

    design = np.column_stack(
        (1.0 / denominator[valid], np.ones(np.count_nonzero(valid)))
    )
    solution, _, rank, _ = np.linalg.lstsq(design, depth[valid], rcond=None)
    if rank < 2:
        raise ValueError("Depth samples do not define both scale and offset.")
    prediction = design @ solution
    rmse = float(np.sqrt(np.mean((prediction - depth[valid]) ** 2)))
    return float(solution[0]), float(solution[1]), rmse


def find_largest_obstacle(
    depth: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    minimum_area_fraction: float = 0.01,
) -> tuple[np.ndarray, Obstacle | None]:
    if depth is None or depth.size == 0 or depth.ndim != 2:
        raise ValueError("Depth must be a non-empty single-channel array.")
    if not 0.0 <= minimum_area_fraction <= 1.0:
        raise ValueError("minimum_area_fraction must be in [0, 1].")
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or min_depth >= max_depth:
        raise ValueError("Depth limits must be finite and min_depth < max_depth.")

    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    mask = np.where(valid, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    contour = max(contours, key=cv2.contourArea)
    image_area = float(mask.shape[0] * mask.shape[1])
    contour_area = float(cv2.contourArea(contour))
    area_fraction = contour_area / image_area
    if area_fraction < minimum_area_fraction:
        return mask, None

    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)
    samples = depth[(contour_mask > 0) & np.isfinite(depth)]
    if samples.size == 0:
        return mask, None
    return mask, Obstacle(
        bounding_box=tuple(int(value) for value in cv2.boundingRect(contour)),
        mean_depth=float(samples.mean()),
        area_fraction=area_fraction,
    )


def disparity_visualization(disparity: np.ndarray) -> np.ndarray:
    valid = np.isfinite(disparity)
    visualization = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(disparity[valid], [2, 98])
        if high > low:
            scaled = np.clip((disparity - low) * 255.0 / (high - low), 0, 255)
            visualization[valid] = scaled[valid].astype(np.uint8)
    return visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a saved depth result from a stereo image pair."
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--already-rectified",
        action="store_true",
        help="Skip the configured rectification maps.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    left = cv2.imread(str(args.left.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(args.right.expanduser().resolve()), cv2.IMREAD_GRAYSCALE)
    _validate_pair(left, right)

    config = load_config(args.config)
    if not args.already_rectified:
        left, right = rectify_pair(left, right, load_rectification_maps(args.maps))

    disparity = compute_disparity(left, right, config)
    depth = disparity_to_depth(disparity, config)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    disparity_image = disparity_visualization(disparity)
    finite_depth = np.isfinite(depth)
    depth_image = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(finite_depth):
        near, far = np.percentile(depth[finite_depth], [2, 98])
        if far > near:
            normalized = np.clip((far - depth) * 255.0 / (far - near), 0, 255)
            depth_image[finite_depth] = normalized[finite_depth].astype(np.uint8)

    for name, image in (
        ("left-rectified.png", left),
        ("right-rectified.png", right),
        ("disparity.png", disparity_image),
        ("depth.png", cv2.applyColorMap(depth_image, cv2.COLORMAP_TURBO)),
    ):
        destination = output_dir / name
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Could not write output image: {destination}")

    print(f"Valid disparity fraction: {np.mean(np.isfinite(disparity)):.4f}")
    print(f"Saved outputs under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

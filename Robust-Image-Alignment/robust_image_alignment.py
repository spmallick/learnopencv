"""Feature matching, robust homography estimation, and explicit alignment rejection.

All images use OpenCV's BGR convention.  Every returned homography maps points
from the moving image into the reference image.  The neural dependencies load
only when an ALIKED-based method is selected, so geometry and SIFT also work
without PyTorch or LightGlue installed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2
import numpy as np


@dataclass
class FeatureSet:
    """Image-local coordinates and descriptors, with optional batched tensors."""

    keypoints: np.ndarray
    descriptors: np.ndarray
    image_shape: tuple[int, int]
    extraction_ms: float
    method: str
    tensors: dict[str, Any] | None = field(default=None, repr=False)


@dataclass
class MatchResult:
    """Paired (x, y) coordinates in the ORIGINAL reference and moving images.

    Scores describe the selected matcher, not probabilities of object identity.
    Nearest-neighbor scores and LightGlue scores are not directly comparable.
    """

    reference_points: np.ndarray
    moving_points: np.ndarray
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    method: str = "provided"
    reference_keypoints: int = 0
    moving_keypoints: int = 0
    extraction_ms: float = 0.0
    matching_ms: float = 0.0

    def __post_init__(self) -> None:
        # An explicit N-by-2 layout avoids OpenCV's alternative N-by-1-by-2 layout.
        self.reference_points = np.asarray(self.reference_points, np.float32).reshape(-1, 2)
        self.moving_points = np.asarray(self.moving_points, np.float32).reshape(-1, 2)
        if len(self.reference_points) != len(self.moving_points):
            raise ValueError("Reference and moving point counts differ.")
        if not np.isfinite(self.reference_points).all() or not np.isfinite(self.moving_points).all():
            raise ValueError("Correspondences must contain only finite coordinates.")
        self.scores = np.asarray(self.scores, np.float32).reshape(-1)
        if len(self.scores) not in (0, len(self.reference_points)):
            raise ValueError("There must be one score per correspondence, or no scores.")


@dataclass(frozen=True)
class AlignmentConfig:
    """Explicit heuristic acceptance gates; tune on validation data, not test pairs.

    Coverage is measured in the reference because a small target may occupy only
    a tiny portion of a cluttered moving image. These defaults are illustrative
    safeguards, not a universal object-identification operating point.
    """

    reprojection_threshold: float = 3.0
    confidence: float = 0.999
    max_iterations: int = 10000
    min_inliers: int = 12
    min_inlier_ratio: float = 0.25
    min_reference_coverage: float = 0.04
    max_median_reprojection_error: float = 3.0
    min_projected_area: float = 16.0
    min_visible_fraction: float = 0.10

    def __post_init__(self) -> None:
        if self.reprojection_threshold <= 0 or self.max_median_reprojection_error <= 0:
            raise ValueError("Pixel error thresholds must be positive.")
        if not 0 < self.confidence < 1 or self.max_iterations < 1 or self.min_inliers < 4:
            raise ValueError("Use 0 < confidence < 1, positive iterations, and at least four inliers.")
        for name in ("min_inlier_ratio", "min_reference_coverage", "min_visible_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.min_projected_area < 0:
            raise ValueError("min_projected_area must be nonnegative.")


@dataclass
class AlignmentResult:
    """A geometric estimate together with its evidence and acceptance decision."""

    H: np.ndarray | None
    inliers: np.ndarray
    accepted: bool
    reasons: list[str]
    projected_reference_quad: np.ndarray | None = None
    median_reprojection_error: float | None = None
    reference_coverage: float = 0.0
    moving_coverage: float = 0.0
    visible_fraction: float = 0.0
    estimation_ms: float = 0.0
    estimator: str = "magsac"
    num_matches: int = 0

    @property
    def num_inliers(self) -> int:
        return int(np.count_nonzero(self.inliers))

    @property
    def inlier_ratio(self) -> float:
        return self.num_inliers / max(self.num_matches, 1)

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-safe record; retain raw correspondences separately."""
        return {
            "accepted": self.accepted,
            "reasons": self.reasons,
            "homography_direction": "moving_to_reference",
            "H": None if self.H is None else self.H.tolist(),
            "num_matches": self.num_matches,
            "num_inliers": self.num_inliers,
            "inlier_ratio": self.inlier_ratio,
            "median_reprojection_error_px": self.median_reprojection_error,
            "reference_inlier_hull_coverage": self.reference_coverage,
            "moving_inlier_hull_coverage": self.moving_coverage,
            "projected_reference_visible_fraction": self.visible_fraction,
            "projected_reference_quad": (
                None if self.projected_reference_quad is None else self.projected_reference_quad.tolist()
            ),
            "estimator": self.estimator,
            "estimation_ms": self.estimation_ms,
        }


def load_image(path: str | Path) -> np.ndarray:
    """Read a three-channel BGR image and report missing or corrupt inputs clearly."""
    image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _check_image(image: np.ndarray) -> None:
    # Keeping one explicit image contract prevents accidental [0,1]/uint8 mixing.
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a nonempty H-by-W-by-3 BGR image.")
    if image.dtype != np.uint8 or min(image.shape[:2]) < 2:
        raise ValueError("Expected a uint8 BGR image at least 2 by 2 pixels.")


class FeatureMatcher:
    """Reusable models and cacheable reference features for three matching routes.

    `lightglue` and `aliked-nn` use exactly the same ALIKED extractor. Call
    `extract` once per image and `match` repeatedly to isolate matcher behavior.
    Model construction and weight downloads are deliberately outside timings.
    """

    def __init__(
        self,
        method: str = "lightglue",
        device: str = "auto",
        max_keypoints: int = 2048,
        resize: int | None = 1024,
        ratio: float = 0.8,
    ) -> None:
        if method not in {"sift", "lightglue", "aliked-nn"}:
            raise ValueError("method must be sift, lightglue, or aliked-nn.")
        if max_keypoints < 4 or not 0 < ratio < 1:
            raise ValueError("Use at least four keypoints and 0 < ratio < 1.")
        if resize is not None and resize < 16:
            raise ValueError("resize must be None or at least 16 pixels.")
        self.method, self.max_keypoints, self.resize, self.ratio = method, max_keypoints, resize, ratio
        self.device = "cpu"
        self._torch = None
        self.extractor = self.matcher = None
        if method == "sift":
            # SIFT uses floating-point descriptors and therefore L2, not Hamming.
            self.extractor = cv2.SIFT_create(nfeatures=max_keypoints)
            return
        try:
            import torch
            from lightglue import ALIKED, LightGlue
        except ImportError as error:
            raise ImportError("Install the pinned LightGlue and PyTorch dependencies first.") from error
        self._torch = torch
        if device == "auto":
            # Select a supported accelerator without assuming CUDA is available.
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable.")
        self.device = device
        # The named extractor variant and matching checkpoint form a tested pair.
        self.extractor = ALIKED(model_name="aliked-n16", max_num_keypoints=max_keypoints).eval().to(device)
        if method == "lightglue":
            self.matcher = LightGlue(features="aliked").eval().to(device)

    def _synchronize(self) -> None:
        # Accelerators are asynchronous; synchronize both sides of timed regions.
        if self.device == "cuda":
            self._torch.cuda.synchronize()
        elif self.device == "mps":
            self._torch.mps.synchronize()

    def extract(self, image_bgr: np.ndarray) -> FeatureSet:
        """Extract features with coordinates mapped back to the original image."""
        _check_image(image_bgr)
        self._synchronize()
        start = perf_counter()
        if self.method == "sift":
            # Match the neural route's longest-side budget while preserving aspect ratio.
            scale = 1.0 if self.resize is None else min(1.0, self.resize / max(image_bgr.shape[:2]))
            work = image_bgr if scale == 1 else cv2.resize(image_bgr, None, fx=scale, fy=scale)
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.extractor.detectAndCompute(gray, None)
            # Tied SIFT responses can exceed nfeatures; enforce the stated budget.
            if len(keypoints) > self.max_keypoints:
                retained = np.argsort([-point.response for point in keypoints], kind="stable")[:self.max_keypoints]
                keypoints = [keypoints[index] for index in retained]
                descriptors = descriptors[retained]
            points = np.array([keypoint.pt for keypoint in keypoints], np.float32).reshape(-1, 2)
            if scale != 1:
                # cv2.resize with fx/fy samples at (destination + 0.5) / scale - 0.5.
                # Rounding the shorter output dimension does not change that sampling scale.
                points = ((points.astype(np.float64) + 0.5) / scale - 0.5).astype(np.float32)
            if descriptors is None:
                descriptors = np.empty((0, 128), np.float32)
            return FeatureSet(points, descriptors, image_bgr.shape[:2], (perf_counter() - start) * 1000, "sift")
        torch = self._torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # The official extractor expects CHW RGB values normalized into [0, 1].
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().to(self.device) / 255.0
        # Cap the longest side without upsampling small references or scene crops.
        resize = None if self.resize is None else min(self.resize, max(image_bgr.shape[:2]))
        with torch.inference_mode():
            tensors = self.extractor.extract(tensor, resize=resize)
        self._synchronize()
        # The wrapper already reverses resizing; do not rescale these points again.
        points = tensors["keypoints"][0].detach().cpu().numpy()
        descriptors = tensors["descriptors"][0].detach().cpu().numpy()
        # Include the host copies used by the baseline and geometric diagnostics.
        elapsed = (perf_counter() - start) * 1000
        return FeatureSet(points, descriptors, image_bgr.shape[:2], elapsed, "aliked-n16", tensors)

    def match(self, reference: FeatureSet, moving: FeatureSet) -> MatchResult:
        """Match cached features; extracted features remain reusable and unmodified."""
        if reference.method != moving.method:
            raise ValueError("Reference and moving descriptors must use the same extractor.")
        if self.method == "sift" and reference.method != "sift":
            raise ValueError("The SIFT baseline expects SIFT features.")
        if self.method != "sift" and reference.method != "aliked-n16":
            raise ValueError("The ALIKED routes expect aliked-n16 features.")
        self._synchronize()
        start = perf_counter()
        indices = np.empty((0, 2), np.int64)
        scores = np.empty(0, np.float32)
        # k=2 ratio matching requires two descriptors in the candidate image.
        if len(reference.keypoints) and len(moving.keypoints) >= 2:
            if self.method == "lightglue":
                if reference.tensors is None or moving.tensors is None:
                    raise ValueError("LightGlue needs the cached batched extractor tensors.")
                with self._torch.inference_mode():
                    output = self.matcher({"image0": reference.tensors, "image1": moving.tensors})
                indices = output["matches"][0].detach().cpu().numpy()
                scores = output["scores"][0].detach().cpu().numpy()
            else:
                # Ratio testing rejects a candidate whose runner-up is almost as good.
                neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference.descriptors, moving.descriptors, k=2)
                selected = [
                    pair for pair in neighbors
                    if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance
                ]
                indices = np.array([(pair[0].queryIdx, pair[0].trainIdx) for pair in selected], np.int64).reshape(-1, 2)
                scores = np.array([
                    1 - pair[0].distance / max(pair[1].distance, 1e-12) for pair in selected
                ], np.float32)
        self._synchronize()
        elapsed = (perf_counter() - start) * 1000
        return MatchResult(
            reference.keypoints[indices[:, 0]], moving.keypoints[indices[:, 1]], scores,
            self.method, len(reference.keypoints), len(moving.keypoints),
            reference.extraction_ms + moving.extraction_ms, elapsed,
        )

    def match_images(self, reference_bgr: np.ndarray, moving_bgr: np.ndarray) -> MatchResult:
        """Convenience route; cache `extract(reference)` yourself for repeated queries."""
        return self.match(self.extract(reference_bgr), self.extract(moving_bgr))


def image_corners(image_shape: tuple[int, ...]) -> np.ndarray:
    """Return pixel-center corners clockwise in image coordinates."""
    height, width = image_shape[:2]
    if height < 2 or width < 2:
        raise ValueError("Image dimensions must be at least two pixels.")
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)


def project_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Project (x, y) coordinates, rejecting points on the projective horizon."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    H = np.asarray(H, np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Expected a finite 3-by-3 homography.")
    coordinates = np.column_stack((points, np.ones(len(points))))
    homogeneous = coordinates @ H.T
    # Compare w with the magnitudes of its three terms so a global rescaling
    # of H does not turn an ordinary point into a spurious horizon rejection.
    denominator_scale = np.abs(coordinates) @ np.abs(H[2, :])
    if np.any(np.abs(homogeneous[:, 2]) <= 1e-10 * denominator_scale):
        raise ValueError("A projected point lies on the projective horizon.")
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(projected).all() or np.any(np.abs(projected) > np.finfo(np.float32).max):
        raise ValueError("Homography projection produced nonfinite coordinates.")
    return projected.astype(np.float32)


def hull_coverage(points: np.ndarray, image_shape: tuple[int, ...]) -> float:
    """Measure spatial support, which match count alone cannot establish."""
    if len(points) < 3:
        return 0.0
    area = cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32)))
    return float(area / ((image_shape[1] - 1) * (image_shape[0] - 1)))


def estimate_alignment(
    matches: MatchResult,
    reference_shape: tuple[int, ...],
    moving_shape: tuple[int, ...],
    config: AlignmentConfig | None = None,
    estimator: str = "magsac",
) -> AlignmentResult:
    """Fit H_moving_to_reference and apply independent, documented rejection gates.

    The reprojection threshold is in ORIGINAL reference-image pixels. MAGSAC++
    accounts for noise-scale uncertainty, but OpenCV still uses this parameter.
    A passed gate is useful evidence; it is not a calibrated identity probability.
    """
    config = config or AlignmentConfig()
    if estimator not in {"magsac", "ransac"}:
        raise ValueError("estimator must be magsac or ransac.")
    result = AlignmentResult(None, np.zeros(len(matches.reference_points), bool), False, [],
                             estimator=estimator, num_matches=len(matches.reference_points))
    # Validate both frame shapes even when the matcher returned no correspondences.
    reference_corners, moving_corners = image_corners(reference_shape), image_corners(moving_shape)
    if result.num_matches < 4:
        result.reasons.append("Fewer than four correspondences.")
        return result
    if (hull_coverage(matches.reference_points, reference_shape) < 1e-6
            or hull_coverage(matches.moving_points, moving_shape) < 1e-6):
        result.reasons.append("Correspondences are collinear or have negligible spatial support.")
        return result
    start = perf_counter()
    try:
        # Source first, destination second: this order fixes the warp direction.
        H, mask = cv2.findHomography(
            matches.moving_points, matches.reference_points,
            method=cv2.USAC_MAGSAC if estimator == "magsac" else cv2.RANSAC,
            ransacReprojThreshold=config.reprojection_threshold,
            maxIters=config.max_iterations, confidence=config.confidence,
        )
    except cv2.error as error:
        result.reasons.append(f"Homography estimation failed: {error.err}")
        return result
    finally:
        result.estimation_ms = (perf_counter() - start) * 1000
    if H is None or mask is None or not np.isfinite(H).all():
        result.reasons.append("No finite homography was estimated.")
        return result
    # Homographies are scale-equivalent; normalization keeps reports readable.
    H = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H / np.linalg.norm(H)
    result.H, result.inliers = H, mask.reshape(-1).astype(bool)
    if result.num_inliers < config.min_inliers:
        result.reasons.append(f"Only {result.num_inliers} inliers; need {config.min_inliers}.")
    if result.inlier_ratio < config.min_inlier_ratio:
        result.reasons.append("Inlier ratio is below the configured minimum.")
    reference_inliers = matches.reference_points[result.inliers]
    moving_inliers = matches.moving_points[result.inliers]
    result.reference_coverage = hull_coverage(reference_inliers, reference_shape)
    result.moving_coverage = hull_coverage(moving_inliers, moving_shape)
    if result.reference_coverage < config.min_reference_coverage:
        result.reasons.append("Inliers cover too little of the reference image.")
    try:
        if result.num_inliers:
            residuals = np.linalg.norm(project_points(moving_inliers, H) - reference_inliers, axis=1)
            result.median_reprojection_error = float(np.median(residuals))
            if result.median_reprojection_error > config.max_median_reprojection_error:
                result.reasons.append("Median inlier reprojection error exceeds the configured limit.")
        H_inverse = np.linalg.inv(H)
        # A target boundary crossing infinity is not a usable physical image region.
        denominators = np.column_stack((reference_corners, np.ones(4))) @ H_inverse[2, :]
        if np.min(denominators) < 0 < np.max(denominators):
            raise ValueError("The projected reference crosses the projective horizon.")
        quad = project_points(reference_corners, H_inverse)
        result.projected_reference_quad = quad
        area = abs(cv2.contourArea(quad))
        if not cv2.isContourConvex(quad) or area < config.min_projected_area:
            result.reasons.append("Projected reference boundary is degenerate or too small.")
        else:
            intersection_area, _ = cv2.intersectConvexConvex(quad, moving_corners)
            result.visible_fraction = float(np.clip(intersection_area / area, 0, 1))
            if result.visible_fraction < config.min_visible_fraction:
                result.reasons.append("Too little of the projected reference lies inside the moving image.")
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        result.reasons.append(f"Invalid projected geometry: {error}")
    result.accepted = not result.reasons
    return result


def warp_alignment(
    moving_bgr: np.ndarray,
    result: AlignmentResult,
    reference_shape: tuple[int, ...],
    require_accepted: bool = True,
    *,
    full_support: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp into reference coordinates and return a uint8 mask (255 inside).

    The default mask follows the nearest source pixel for display overlays.
    Set full_support=True for intensity comparisons to exclude any pixel whose
    bilinear interpolation uses a nonzero contribution from the constant border.
    """
    _check_image(moving_bgr)
    if result.H is None or (require_accepted and not result.accepted):
        raise ValueError("Alignment was rejected; inspect the report before warping.")
    height, width = reference_shape[:2]
    aligned = cv2.warpPerspective(moving_bgr, result.H, (width, height), flags=cv2.INTER_LINEAR)
    if full_support:
        # A float mask preserves fractional weights, including tiny corner contributions.
        source_ones = np.ones(moving_bgr.shape[:2], np.float32)
        weight = cv2.warpPerspective(source_ones, result.H, (width, height), flags=cv2.INTER_LINEAR)
        valid = cv2.compare(weight, 1.0, cv2.CMP_EQ)
    else:
        # Source colors, including genuine black pixels, do not affect validity.
        source_mask = np.full(moving_bgr.shape[:2], 255, np.uint8)
        valid = cv2.warpPerspective(source_mask, result.H, (width, height), flags=cv2.INTER_NEAREST)
    return aligned, valid


def refine_ecc(
    reference_bgr: np.ndarray,
    moving_bgr: np.ndarray,
    H_moving_to_reference: np.ndarray,
    iterations: int = 100,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Optionally refine a reasonable homography; ECC may fail or worsen a result.

    OpenCV ECC estimates template-to-input coordinates. Its initialization must
    therefore be the INVERSE of this module's moving-to-reference convention.
    Assess held-out geometric error before accepting the returned refinement.
    """
    _check_image(reference_bgr)
    _check_image(moving_bgr)
    if iterations < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("ECC iterations and epsilon must be positive.")
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    moving_gray = cv2.cvtColor(moving_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    # Remove arbitrary projective scale before inversion to handle equivalent inputs alike.
    initial = np.asarray(H_moving_to_reference, dtype=np.float64)
    if initial.shape != (3, 3) or not np.isfinite(initial).all() or not np.any(initial):
        raise ValueError("ECC needs a finite, nonzero 3-by-3 initial homography.")
    inverse = np.linalg.inv(initial / np.max(np.abs(initial)))
    # The ECC homography Jacobian fixes the bottom-right coefficient to one.
    # Inversion generally changes it, even when the incoming matrix had H[2, 2] == 1.
    if not np.isfinite(inverse).all() or abs(inverse[2, 2]) <= 1e-12 * np.max(np.abs(inverse)):
        raise ValueError("ECC cannot normalize its reference-to-moving initialization at [2, 2].")
    initial_reference_to_moving = (inverse / inverse[2, 2]).astype(np.float32)
    # Restrict optimization to source pixels that the initial estimate places
    # inside the reference; scene clutter and black padding are not evidence.
    input_mask = cv2.warpPerspective(
        np.full(reference_bgr.shape[:2], 255, np.uint8), initial_reference_to_moving,
        (moving_bgr.shape[1], moving_bgr.shape[0]), flags=cv2.INTER_NEAREST,
    )
    # Exclude the Gaussian filter's support near the valid-overlap boundary.
    input_mask = cv2.erode(input_mask, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    if np.count_nonzero(input_mask) < 64:
        raise ValueError("Too little valid overlap for ECC refinement.")
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    correlation, reference_to_moving = cv2.findTransformECC(
        reference_gray, moving_gray, initial_reference_to_moving,
        cv2.MOTION_HOMOGRAPHY, criteria, input_mask, 5,
    )
    # Return to the public direction in double precision. The inverse may have H[2, 2] == 0.
    H = np.linalg.inv(reference_to_moving.astype(np.float64))
    if not np.isfinite(H).all() or not np.isfinite(correlation):
        raise ValueError("ECC returned a nonfinite refinement.")
    max_entry = np.max(np.abs(H))
    scale = H[2, 2] if abs(H[2, 2]) > 1e-12 * max_entry else max_entry
    return H / scale, float(correlation)


def draw_matches(
    reference_bgr: np.ndarray,
    moving_bgr: np.ndarray,
    matches: MatchResult,
    result: AlignmentResult | None = None,
    max_draw: int = 100,
) -> np.ndarray:
    """Draw a reproducible subset of matches; green=inlier, red=outlier."""
    if max_draw < 1:
        raise ValueError("max_draw must be positive.")
    height = max(reference_bgr.shape[0], moving_bgr.shape[0])
    offset = reference_bgr.shape[1]
    canvas = np.zeros((height, offset + moving_bgr.shape[1], 3), np.uint8)
    canvas[:reference_bgr.shape[0], :offset] = reference_bgr
    canvas[:moving_bgr.shape[0], offset:] = moving_bgr
    selected = np.linspace(
        0, len(matches.reference_points) - 1, min(max_draw, len(matches.reference_points)), dtype=int,
    )
    for index in selected:
        point0 = tuple(np.rint(matches.reference_points[index]).astype(int))
        point1 = tuple(np.rint(matches.moving_points[index] + [offset, 0]).astype(int))
        color = (40, 200, 40) if result is None or result.inliers[index] else (40, 40, 220)
        cv2.line(canvas, point0, point1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point1, 3, color, -1, cv2.LINE_AA)
    return canvas


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference image or planar target crop.")
    parser.add_argument("--moving", type=Path, required=True,
                        help="Image to align, or scene in which to locate the target.")
    parser.add_argument("--output-dir", type=Path, default=Path("alignment-output"))
    parser.add_argument("--method", choices=("sift", "lightglue", "aliked-nn"), default="lightglue")
    parser.add_argument("--estimator", choices=("magsac", "ransac"), default="magsac")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--resize", type=int, default=1024, help="Longest-side cap; use 0 to preserve full resolution.")
    parser.add_argument("--threshold", type=float, default=3.0, help="Error threshold in original reference pixels.")
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--no-display", action="store_true", help="Headless mode (also the default).")
    parser.add_argument("--validate", action="store_true", help="Exit nonzero unless the alignment is accepted.")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference, moving = load_image(args.reference), load_image(args.moving)
    matcher = FeatureMatcher(args.method, args.device, args.max_keypoints, args.resize or None)
    matches = matcher.match_images(reference, moving)
    config = AlignmentConfig(reprojection_threshold=args.threshold, min_inliers=args.min_inliers)
    result = estimate_alignment(matches, reference.shape, moving.shape, config, args.estimator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visualization = draw_matches(reference, moving, matches, result)
    _write_image(args.output_dir / "matches.jpg", visualization)
    if result.accepted:
        aligned, mask = warp_alignment(moving, result, reference.shape)
        overlay = cv2.addWeighted(reference, 0.5, aligned, 0.5, 0)
        # Invalid border pixels should not darken the reference in an inspection overlay.
        overlay[mask == 0] = reference[mask == 0]
        boundary = moving.copy()
        cv2.polylines(boundary, [np.rint(result.projected_reference_quad).astype(np.int32)],
                      True, (40, 220, 40), 3, cv2.LINE_AA)
        for name, image in (("aligned.png", aligned), ("valid_mask.png", mask),
                            ("overlay.jpg", overlay), ("reference_boundary.jpg", boundary)):
            _write_image(args.output_dir / name, image)
    else:
        # A reused directory must not present a previous accepted warp as this
        # rejected run's result. Remove only the four outputs owned by this CLI.
        for name in ("aligned.png", "valid_mask.png", "overlay.jpg", "reference_boundary.jpg"):
            (args.output_dir / name).unlink(missing_ok=True)
    report = result.to_dict()
    report.update({"opencv_version": cv2.__version__, "method": args.method, "device": matcher.device,
                   "reference_keypoints": matches.reference_keypoints, "moving_keypoints": matches.moving_keypoints,
                   "extraction_ms": matches.extraction_ms, "matching_ms": matches.matching_ms,
                   "configuration": asdict(config)})
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (ValueError, OSError, ImportError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.validate:
        if not report["accepted"]:
            print("VALIDATION FAILED: alignment rejected", file=sys.stderr)
            return 3
        print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

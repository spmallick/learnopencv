#!/usr/bin/env python3
"""Shared helpers for the ECC image-alignment examples."""

from __future__ import annotations

import cv2
import numpy as np


MOTION_MODELS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


def gradient(image: np.ndarray) -> np.ndarray:
    """Return a float gradient-magnitude proxy suited to ECC alignment."""
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.addWeighted(np.abs(grad_x), 0.5, np.abs(grad_y), 0.5, 0.0)


def identity_warp(motion_model: int) -> np.ndarray:
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, dtype=np.float32)
    return np.eye(2, 3, dtype=np.float32)


def align_image(
    template: np.ndarray,
    moving: np.ndarray,
    motion_model: int,
    iterations: int,
    epsilon: float,
    use_gradient: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate an ECC warp and return correlation, warp, and aligned image."""
    if template.shape[:2] != moving.shape[:2]:
        moving = cv2.resize(
            moving, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA
        )
    template_gray = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if template.ndim == 3
        else template
    )
    moving_gray = (
        cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY) if moving.ndim == 3 else moving
    )
    if use_gradient:
        template_input = gradient(template_gray)
        moving_input = gradient(moving_gray)
    else:
        template_input = template_gray.astype(np.float32)
        moving_input = moving_gray.astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        iterations,
        epsilon,
    )
    correlation, warp = cv2.findTransformECC(
        template_input,
        moving_input,
        identity_warp(motion_model),
        motion_model,
        criteria,
    )
    size = (template.shape[1], template.shape[0])
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    if motion_model == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving, warp, size, flags=flags)
    else:
        aligned = cv2.warpAffine(moving, warp, size, flags=flags)
    return float(correlation), warp, aligned


def mean_absolute_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(first, second)))

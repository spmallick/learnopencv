#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15
MIN_HOMOGRAPHY_MATCHES = 4


class AlignmentStats(NamedTuple):
    input_keypoints: int
    reference_keypoints: int
    total_matches: int
    retained_matches: int
    inlier_matches: int


def _write_image(filename, image):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(filename), image):
        raise OSError(f"Could not write image: {filename}")


def _match_sort_key(match, keypoints1, keypoints2):
    point1 = keypoints1[match.queryIdx].pt
    point2 = keypoints2[match.trainIdx].pt
    return (
        float(match.distance),
        float(point1[1]),
        float(point1[0]),
        float(point2[1]),
        float(point2[0]),
        int(match.queryIdx),
        int(match.trainIdx),
        int(match.imgIdx),
    )


def _align_images_impl(image, reference, matches_filename=None):
    if image is None or image.size == 0:
        raise ValueError("The image to align is empty")
    if reference is None or reference.size == 0:
        raise ValueError("The reference image is empty")

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(image_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_gray, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("ORB could not find features in both images")

    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
    matches = list(matcher.match(descriptors1, descriptors2))

    if len(matches) < MIN_HOMOGRAPHY_MATCHES:
        raise ValueError(
            f"At least {MIN_HOMOGRAPHY_MATCHES} feature matches are required; "
            f"found {len(matches)}"
        )

    matches.sort(key=lambda match: _match_sort_key(match, keypoints1, keypoints2))
    nominal_match_count = max(
        MIN_HOMOGRAPHY_MATCHES,
        int(len(matches) * GOOD_MATCH_PERCENT),
    )
    cutoff_distance = matches[nominal_match_count - 1].distance
    good_matches = [
        match for match in matches if match.distance <= cutoff_distance
    ]

    if matches_filename is not None:
        matches_image = cv2.drawMatches(
            image,
            keypoints1,
            reference,
            keypoints2,
            good_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        _write_image(matches_filename, matches_image)

    points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches])

    homography, inlier_mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Could not estimate a valid homography")
    if inlier_mask is None:
        raise ValueError("Could not determine homography inliers")

    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(image, homography, (width, height))
    stats = AlignmentStats(
        input_keypoints=len(keypoints1),
        reference_keypoints=len(keypoints2),
        total_matches=len(matches),
        retained_matches=len(good_matches),
        inlier_matches=int(np.count_nonzero(inlier_mask)),
    )
    return aligned, homography, stats


def align_images(image, reference, matches_filename=None):
    """Align ``image`` to ``reference`` and return the image and homography."""
    aligned, homography, _ = _align_images_impl(
        image,
        reference,
        matches_filename,
    )
    return aligned, homography


def alignImages(image, reference):
    """Backward-compatible wrapper used by the original LearnOpenCV article."""
    return align_images(image, reference, "matches.jpg")


def _read_image(filename, description):
    image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {description}: {filename}")
    return image


def validate_alignment(image, reference, aligned):
    if aligned.shape != reference.shape:
        raise ValueError("Aligned image dimensions do not match the reference")

    resized_input = cv2.resize(image, (reference.shape[1], reference.shape[0]))
    before = float(np.mean(cv2.absdiff(resized_input, reference)))
    after = float(np.mean(cv2.absdiff(aligned, reference)))
    print(f"Alignment MAE before: {before:.6f}")
    print(f"Alignment MAE after: {after:.6f}")
    if (
        not np.isfinite((before, after)).all()
        or before <= 0
        or after >= before * 0.4
    ):
        raise ValueError("Alignment did not improve mean absolute error by at least 60%")


def validate_written_outputs(
    aligned_path,
    matches_path,
    aligned_shape,
    matches_shape,
):
    aligned = _read_image(aligned_path, "written aligned image")
    matches = _read_image(matches_path, "written feature matches image")
    if aligned.shape != aligned_shape:
        raise ValueError(
            "Written aligned image dimensions do not match the reference"
        )
    if matches.shape != matches_shape:
        raise ValueError(
            "Written feature matches dimensions are not as expected"
        )


def parse_args(argv=None):
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Feature-based image alignment with OpenCV"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_dir / "scanned-form.jpg",
        help="image to align",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=project_dir / "form.jpg",
        help="reference image",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="accepted for headless workflows",
    )
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def run(args):
    output_dir = args.output_dir.resolve()
    aligned_path = output_dir / "aligned.jpg"
    matches_path = output_dir / "matches.jpg"
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Reading reference image: {args.reference}")
    reference = _read_image(args.reference, "reference image")
    print(f"Reading image to align: {args.input}")
    image = _read_image(args.input, "input image")

    print("Aligning images ...")
    aligned, homography, stats = _align_images_impl(
        image,
        reference,
        matches_path,
    )
    print(
        "Detected keypoints: "
        f"input={stats.input_keypoints}, reference={stats.reference_keypoints}"
    )
    print(
        "Feature matches: "
        f"total={stats.total_matches}, retained={stats.retained_matches}, "
        f"inliers={stats.inlier_matches}"
    )
    if args.validate:
        validate_alignment(image, reference, aligned)
    _write_image(aligned_path, aligned)
    if args.validate:
        matches_shape = (
            max(image.shape[0], reference.shape[0]),
            image.shape[1] + reference.shape[1],
            3,
        )
        validate_written_outputs(
            aligned_path,
            matches_path,
            reference.shape,
            matches_shape,
        )

    print(f"Saved aligned image: {aligned_path}")
    print(f"Saved feature matches: {matches_path}")
    print("Estimated homography:\n", homography)
    if args.validate:
        print("Alignment validation passed")
    return aligned, homography


def main(argv=None):
    try:
        run(parse_args(argv))
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

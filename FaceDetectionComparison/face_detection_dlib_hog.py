#!/usr/bin/env python3
"""Run the optional historical dlib HOG face detector."""

from __future__ import annotations

import sys

import cv2

from face_detection import DlibHogDetector
from historical_detector_cli import parse_historical_args, run_historical


def main(argv: list[str] | None = None) -> int:
    """Import dlib only when this explicit historical example is requested."""

    try:
        args = parse_historical_args(__doc__ or "dlib HOG face detection", argv)
        run_historical(args, DlibHogDetector(), "dlib-hog")
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Small optional OpenCV window wrapper used by the SLAM command line."""

from __future__ import annotations

import cv2
import numpy as np


class Display:
    """Show resized frames while keeping all rendering optional for tests."""

    def __init__(self, width: int, height: int, title: str = "Monocular SLAM"):
        self.width = width
        self.height = height
        self.title = title

    def show(self, image: np.ndarray, delay_milliseconds: int = 1) -> int:
        """Resize, display, and return OpenCV's extended keyboard code."""

        resized = cv2.resize(image, (self.width, self.height))
        cv2.imshow(self.title, resized)
        return cv2.waitKeyEx(delay_milliseconds)

    @staticmethod
    def close() -> None:
        """Close windows created by this process."""

        cv2.destroyAllWindows()

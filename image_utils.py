"""Image processing utilities module using OpenCV and Python."""
import cv2
import numpy as np
from typing import Tuple, Optional

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from file path."""
    try:
        image = cv2.imread(image_path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize image to specified dimensions."""
    return cv2.resize(image, (width, height))

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Apply Gaussian blur to image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def detect_edges(image: np.ndarray, threshold1: int = 100, threshold2: int = 200) -> np.ndarray:
    """Detect edges using Canny edge detection."""
    gray = convert_to_grayscale(image)
    return cv2.Canny(gray, threshold1, threshold2)

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to improve contrast."""
    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = cv2.equalizeHist(v)
        hsv_eq = cv2.merge((h, s, v_eq))
        return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)

def detect_contours(image: np.ndarray) -> Tuple[np.ndarray, list]:
    """Detect contours in image."""
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return binary, contours

def draw_contours(image: np.ndarray, contours: list) -> np.ndarray:
    """Draw contours on image."""
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result

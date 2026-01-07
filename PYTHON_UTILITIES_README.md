# Python Utilities for Learn OpenCV

## Overview

This contribution adds a collection of Python utilities and interactive Streamlit applications to enhance the Learn OpenCV project. The utilities provide convenient wrappers around common OpenCV operations and demonstrate best practices for image processing in Python.

## Files Included

### 1. `image_utils.py`
A comprehensive utility module containing reusable functions for image processing:
- **load_image()**: Load images from file paths with error handling
- **resize_image()**: Resize images to specified dimensions
- **convert_to_grayscale()**: Convert images to grayscale format
- **apply_gaussian_blur()**: Apply Gaussian blur filter for smoothing
- **detect_edges()**: Canny edge detection for contour identification
- **apply_histogram_equalization()**: Enhance image contrast
- **detect_contours()**: Find contours in images
- **draw_contours()**: Visualize detected contours on images

### 2. `image_processor_app.py`
An interactive Streamlit web application for image processing:
- Upload images via web interface
- Apply various filters in real-time
- Side-by-side comparison of original and processed images
- Support for multiple processing modes:
  - Grayscale conversion
  - Edge detection (Canny)
  - Gaussian blur
  - Histogram equalization

### 3. `requirements.txt`
Python dependencies for all utilities:
- OpenCV (opencv-python)
- NumPy for numerical operations
- Streamlit for web interface
- Pandas for data handling
- Matplotlib for visualization
- Scikit-learn for ML utilities
- TensorFlow for deep learning examples
- Pillow for image operations
- SciPy for scientific computing

## Installation

```bash
# Clone the repository
git clone https://github.com/Vinay21rout/learnopencv.git
cd learnopencv

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Using Image Utilities

```python
from image_utils import load_image, convert_to_grayscale, detect_edges

# Load and process image
image = load_image('path/to/image.jpg')
if image is not None:
    gray = convert_to_grayscale(image)
    edges = detect_edges(gray)
```

### Running the Streamlit App

```bash
streamlit run image_processor_app.py
```

Then open your browser and navigate to `http://localhost:8501`

## Features

- **Type hints**: All functions include Python type hints for better IDE support
- **Error handling**: Robust error handling in utility functions
- **Documentation**: Comprehensive docstrings for all functions
- **Interactive UI**: Streamlit provides an intuitive web interface
- **Real-time processing**: Instant feedback on image transformations

## Future Enhancements

- Add more advanced image processing filters
- Support for batch processing
- ML-based image classification integration
- Video processing capabilities
- Performance optimization for large images

## Contribution

These utilities are designed to be educational and can serve as a foundation for more complex image processing projects. Feel free to extend and modify these utilities for your specific use cases.

## License

This contribution follows the same license as the Learn OpenCV repository.

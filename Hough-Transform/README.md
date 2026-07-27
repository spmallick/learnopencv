# [Hough Transform with OpenCV in Python and C++](https://learnopencv.com/hough-transform-with-opencv-c-python/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2019/03/04095601/hough-transform-with-opencv.png" alt="Hough line and circle detection with OpenCV" width="640">](https://learnopencv.com/hough-transform-with-opencv-c-python/)

This companion project detects road-line segments with the probabilistic Hough
transform and detects eye boundaries with OpenCV's gradient circle transform.
The Python and C++ examples share the same preprocessing and parameters:

- grayscale conversion and Gaussian blur before Canny line edges;
- `HoughLinesP` for finite line segments;
- median blur before `HoughCircles`;
- deterministic sorting, output filenames, and validation metrics.

Hough methods can tolerate partial or broken edge evidence, but they are still
sensitive to edge quality, clutter, accumulator thresholds, and expected shape
sizes.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/hough-transform-opencv-2026.07.26/Hough-Transform-2026.07.26.zip)

## Requirements

- Python 3.10 or newer
- NumPy 1.23 or newer
- OpenCV 4.x or OpenCV 5.x
- CMake 3.16 or newer and a C++17 compiler

```bash
python3 -m pip install -r requirements.txt
```

## Python usage

Detect line segments and write the edge/result images:

```bash
python3 hough_lines.py lanes.jpg \
  --no-display \
  --output-dir outputs
```

Detect circles:

```bash
python3 hough_circles.py brown-eyes.jpg \
  --no-display \
  --output-dir outputs
```

The line program exposes Canny thresholds, accumulator threshold, minimum line
length, and maximum gap. The circle program exposes `dp`, minimum center
distance, `param1`, `param2`, and radius bounds. Run either command with
`--help` for the complete interface. Omit `--no-display` to open result
windows.

## C++ build and usage

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

./build/hough_lines lanes.jpg \
  --no-display \
  --output-dir outputs

./build/hough_circles brown-eyes.jpg \
  --no-display \
  --output-dir outputs
```

Use `-DOpenCV_DIR=/path/to/lib/cmake/opencv4` or the corresponding OpenCV 5
configuration directory to select a specific installation.

## Parameters that matter

For `HoughLinesP`, the accumulator threshold controls the minimum number of
votes, `minLineLength` rejects short segments, and `maxLineGap` joins nearby
collinear evidence.

For `HoughCircles` with `HOUGH_GRADIENT`:

- `param1` is the upper Canny threshold used internally;
- `param2` is the center-detection accumulator threshold;
- lower `param2` values usually return more candidates and false positives;
- realistic radius bounds reduce both compute and ambiguity.

## Validation and tests

Validation combines the bundled photographs with generated images containing a
known horizontal line, diagonal line, and circle. It checks semantic geometry
instead of depending on an undocumented order from OpenCV.

```bash
python3 hough_lines.py --validate --no-display
python3 hough_circles.py --validate --no-display
python3 -m unittest discover -s tests -v

./build/hough_lines --validate --no-display
./build/hough_circles --validate --no-display
ctest --test-dir build --output-on-failure
```

The tests also cover empty detections, invalid parameters, output readability,
missing inputs, and execution from an unrelated working directory.

## Project layout

```text
Hough-Transform/
├── CMakeLists.txt
├── README.md
├── brown-eyes.jpg
├── hough_circles.cpp
├── hough_circles.py
├── hough_common.hpp
├── hough_lines.cpp
├── hough_lines.py
├── hough_utils.py
├── lanes.jpg
├── requirements.txt
└── tests/
    └── test_hough.py
```

---

<p align="center">
  <a href="https://bigvision.ai/">
    <img src="https://bigvision.ai/logos/logo.png" alt="BigVision.AI" width="300">
  </a>
</p>

<h2 align="center">Build Production-Ready Computer Vision &amp; AI Solutions</h2>

<p align="center">
  LearnOpenCV is maintained by <a href="https://bigvision.ai/"><strong>BigVision.AI</strong></a>, a computer vision and AI consulting company. We help organizations design, build, optimize, and deploy production-ready AI solutions. Our team has deep expertise in computer vision, deep learning, multimodal AI, and edge deployment, with experience solving complex technical challenges across industries.
</p>

<p align="center">
  Have a project in mind? Talk with our expert AI solution builders.
</p>

<p align="center">
  <a href="https://bigvision.ai/expert-ai-solution-builders?utm_source=locv-github">
    <img src="https://img.shields.io/badge/Get%20in%20Touch-087EA4?style=for-the-badge" alt="Get in Touch with BigVision.AI">
  </a>
</p>

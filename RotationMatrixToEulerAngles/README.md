# [Rotation Matrix to Euler Angles](https://learnopencv.com/rotation-matrix-to-euler-angles/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2017/12/04104052/favicon.png" alt="Rotation Matrix to Euler Angles tutorial" width="640">](https://learnopencv.com/rotation-matrix-to-euler-angles/)

This companion project converts between 3×3 rotation matrices and XYZ
Tait–Bryan angles in Python and C++. The implementation uses active,
right-handed rotations of column vectors:

```text
R = Rz(z) @ Ry(y) @ Rx(x)
```

Angles are ordered `[x, y, z]`. They are radians by default, and the CLI can
read and display degrees. Because angle triples are not unique, validation
compares reconstructed rotation matrices instead of comparing angles directly.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/rotation-matrix-euler-opencv-2026.07.26/RotationMatrixToEulerAngles-2026.07.26.zip)

## Requirements

- Python 3.10 or newer
- NumPy 1.23 or newer
- CMake 3.16 or newer
- A C++17 compiler
- OpenCV 4.x or OpenCV 5.x with the `core` module

Install the Python dependency:

```bash
python3 -m pip install -r requirements.txt
```

## Python usage

Run the deterministic default example:

```bash
python3 rotm2euler.py --degrees
```

Convert an angle vector:

```bash
python3 rotm2euler.py --degrees --angles 20 -35 70
```

Convert a row-major rotation matrix:

```bash
python3 rotm2euler.py --matrix 1 0 0 0 1 0 0 0 1
```

Run the built-in validation:

```bash
python3 rotm2euler.py --validate
```

## C++ build and usage

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/rotm2euler --degrees --angles 20 -35 70
./build/rotm2euler --validate
```

Use `-DOpenCV_DIR=/path/to/lib/cmake/opencv4` or the corresponding OpenCV 5
configuration directory when selecting a non-default installation.

## Tests

The tests cover known rotations, exact and near gimbal lock, reflections,
malformed matrices, deterministic random round trips, and execution from an
unrelated working directory.

```bash
python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
```

## Project layout

```text
RotationMatrixToEulerAngles/
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── rotm2euler.cpp
├── rotm2euler.py
└── tests/
    └── test_rotm2euler.py
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

# [Filling Holes in an Image Using OpenCV (Python and C++)](https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/)

<p align="center">
  <a href="https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/">
    <img src="nickel.jpg" alt="Nickel image used to demonstrate filling holes in a binary mask" width="600">
  </a>
</p>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download the tested code" width="200">](https://github.com/spmallick/learnopencv/releases/download/fill-holes-opencv-2026.07.27/Holes-2026.07.27.zip)

This project thresholds an image, flood-fills only the exterior background, and
combines the result with the original binary mask. The refreshed Python and C++
programs are headless by default, write every intermediate stage, validate
inputs, and use a padded border so foreground touching an image edge does not
invalidate the flood-fill seed.

## Requirements

- Python 3.10 or newer
- NumPy 2.0 or newer
- OpenCV 4.8 or newer, including OpenCV 5.x
- CMake 3.16 and a C++17 compiler for the native example

The tested environments were Python 3.14.3 with OpenCV-Python 4.13.0, exact
native OpenCV 4.13.0 with AppleClang 21, and the official OpenCV 5.0.0 build in both
Python and C++.

## Python

```bash
python -m pip install -r requirements.txt
python imfill.py nickel.jpg --output-dir output
```

Add `--display` only on a machine with a graphical desktop. The output directory
contains the thresholded mask, flooded exterior, isolated holes, and final
filled mask.

## C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/imfill nickel.jpg --output-dir output-cpp
```

To select a specific OpenCV installation, pass its configuration directory:

```bash
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5
```

## Tests

```bash
python -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
```

The Python suite covers a synthetic hole, an edge-touching object, invalid
non-binary input, and the bundled nickel regression. CTest runs the compiled
headless pipeline. The same four Python tests and one C++ smoke test passed with
OpenCV 4 and exact OpenCV 5.0.0.

## Project layout

```text
Holes/
├── CMakeLists.txt
├── README.md
├── imfill.cpp
├── imfill.py
├── nickel.jpg
├── requirements.txt
└── tests/
    └── test_imfill.py
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

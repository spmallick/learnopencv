# [Image Quality Assessment With BRISQUE in OpenCV](https://learnopencv.com/image-quality-assessment-brisque/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2018/06/04095951/brisque-iqa-feature-nr-e1529032216542.png" alt="BRISQUE no-reference image quality assessment" width="651">](https://learnopencv.com/image-quality-assessment-brisque/)

This directory contains current Python and C++ examples for no-reference image
quality assessment with OpenCV's maintained
[`QualityBRISQUE`](https://docs.opencv.org/4.x/d8/d99/classcv_1_1quality_1_1QualityBRISQUE.html)
implementation. Lower scores generally indicate fewer natural-image
distortions.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download code" width="200">](https://github.com/spmallick/learnopencv/releases/download/brisque-v2026.07.26/ImageMetrics-2026.07.26.zip)

## What changed

The examples now call OpenCV's quality module directly. They no longer require
Python 2, SciPy, a separately compiled LIBSVM library, or checked-in platform
binaries. The pretrained model and feature ranges are the files distributed
with OpenCV Contrib 5.0.0; their source and checksums are recorded in
[`models/README.md`](models/README.md).

## Requirements

- Python 3.10–3.14
- NumPy 1.26–2.x
- OpenCV Contrib 4.8–5.x with the `quality` module
- CMake 3.16 or newer and a C++17 compiler for the C++ example

The examples were verified with Python 3.14, OpenCV 4.12, 4.13, and 5.0.0,
NumPy 2.4, CMake 3.29, and Apple Clang 21. The declared Python requirements
also passed all four tests in an isolated environment containing only
`opencv-contrib-python` 5.0.0.93 (not the standard wheel).

The standard `opencv-python` wheel does not include the quality module. For
Python, install `opencv-contrib-python` as shown below.

## Python

Install the dependencies:

```bash
python -m pip install -r ImageMetrics/Python/requirements.txt
```

Score the included sample image from the repository root:

```bash
python ImageMetrics/Python/brisquequality.py \
  ImageMetrics/Images/original-scaled-image.jpg
```

Expected output with OpenCV 4.13:

```text
BRISQUE score: 20.2789
```

The script resolves the default model and range files relative to its own
location, so it works from any current directory. Use `--model` and `--range`
to evaluate with another compatible OpenCV BRISQUE model.

## C++

The C++ OpenCV installation must include the `quality` module from
OpenCV Contrib. Configure an out-of-source build:

```bash
cmake -S ImageMetrics/C++ -B ImageMetrics/C++/build
cmake --build ImageMetrics/C++/build --parallel
```

Run the sample:

```bash
./ImageMetrics/C++/build/brisquequality \
  ImageMetrics/Images/original-scaled-image.jpg
```

If CMake cannot locate OpenCV, pass its package directory explicitly:

```bash
cmake -S ImageMetrics/C++ -B ImageMetrics/C++/build \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

## Tests

Run the Python regression tests:

```bash
python -m pip install -r ImageMetrics/Python/requirements-dev.txt
python -m pytest ImageMetrics/tests
```

Run the C++ sample-score regression:

```bash
ctest --test-dir ImageMetrics/C++/build --output-on-failure
```

The tests check the pinned sample score, confirm that strong blur receives a
worse score, validate missing-file handling, and exercise the default model
lookup from a different working directory.


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

# [Shape Matching Using Hu Moments in OpenCV](https://learnopencv.com/shape-matching-using-hu-moments-c-python/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2018/12/04095653/Hu-Moments.jpg" alt="Shape matching using Hu Moments in OpenCV" width="650">](https://learnopencv.com/shape-matching-using-hu-moments-c-python/)

This directory contains tested Python and C++ examples for calculating the
seven Hu moment invariants and comparing binary shapes with
[`matchShapes`](https://docs.opencv.org/5.0/main_modules/geometry_shape.html).

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download code" width="200">](https://github.com/spmallick/learnopencv/releases/download/hu-moments-v2026.07.26/HuMoments-2026.07.26.zip)

## Requirements

- Python 3.10–3.14
- OpenCV 4.8–5.x
- NumPy 1.26–2.x
- CMake 3.16 or newer and a C++17 compiler for the C++ examples

The examples were verified with Python 3.14, OpenCV 4.12, 4.13, and 5.0.0,
NumPy 2.4, CMake 3.29, and Apple Clang 21. The C++ build automatically links
the `geometry` module introduced by OpenCV 5.

## Python

Install the Python dependencies:

```bash
python -m pip install -r HuMoments/requirements.txt
```

From the repository root, calculate signed log-transformed Hu moments:

```bash
python HuMoments/HuMoments.py HuMoments/images/*.png
```

Print the untransformed values with `--raw`, or change the binary threshold
with `--threshold`.

Compare the provided `S` shape with another letter and a transformed `S`:

```bash
python HuMoments/shapeMatcher.py
```

The script resolves its sample images relative to its own location, so the
default command works from any current directory. It also accepts three
explicit image paths:

```bash
python HuMoments/shapeMatcher.py REFERENCE DIFFERENT TRANSFORMED
```

## C++

Configure an out-of-source build:

```bash
cmake -S HuMoments -B HuMoments/build
cmake --build HuMoments/build --parallel
```

Run the examples:

```bash
./HuMoments/build/HuMoments HuMoments/images/*.png
./HuMoments/build/shapeMatcher
```

If CMake cannot locate OpenCV, pass its package directory explicitly:

```bash
cmake -S HuMoments -B HuMoments/build -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

## Tests

Run the Python regression tests:

```bash
python -m pip install -r HuMoments/requirements-dev.txt
python -m pytest HuMoments/tests
```

Run the C++ smoke and numeric-regression tests:

```bash
ctest --test-dir HuMoments/build --output-on-failure
```

The tests verify the known `K0` Hu moments, the zero-safe log transform, and
that the transformed `S4` image is closer to `S0` than the different `K0`
shape.


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

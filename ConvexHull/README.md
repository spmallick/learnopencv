# [Convex Hull Using OpenCV in Python and C++](https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/)

This directory contains the tested Python and C++ companion code for the
[LearnOpenCV article](https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/).

<a href="https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/"><img src="https://cdn.learnopencv.com/wp-content/uploads/2018/08/04095813/Foundation-768x576.jpg" alt="Convex Hull using OpenCV in C++ and Python" width="900"></a>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://github.com/spmallick/learnopencv/releases/download/convex-hull-v2026.07.26/ConvexHull-2026.07.26.zip)

The examples use the current two-result `findContours` API, validate the input,
write an output image in headless environments, and open windows only when
`--display` is requested.

## Requirements

- Python 3.10 or newer
- OpenCV 4.8 or newer
- CMake 3.16 or newer and a C++17 compiler for the C++ example

```bash
python3 -m pip install -r requirements.txt
```

## Python

Run the bundled sample with the default threshold:

```bash
python3 example.py
```

All important inputs are configurable:

```bash
python3 example.py \
  --input sample.jpg \
  --output convex-hull-output.jpg \
  --threshold 200
```

Add `--display` when a desktop GUI is available.

## C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/ConvexHull \
  --input=sample.jpg \
  --output=convex-hull-output-cpp.jpg \
  --threshold=200
```

## Tests

```bash
python3 -m unittest discover -s tests -v
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

The Python tests verify a synthetic concave contour, invalid threshold handling,
and the bundled sample. CTest runs the compiled C++ program headlessly and
checks that it detects contours and writes its result.

The compatibility matrix passed Python 3/3 and CTest 1/1 against the official
OpenCV 5.0.0 tag. Local OpenCV 4 validation passed Python 3/3 with 4.13.0 and
CTest 1/1 with native 4.12.0.

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

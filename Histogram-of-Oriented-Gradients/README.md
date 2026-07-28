# [Histogram of Oriented Gradients explained using OpenCV](https://learnopencv.com/histogram-of-oriented-gradients/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2016/12/04100848/histogram-of-oriented-gradients.jpg" alt="A person image with its Histogram of Oriented Gradients visualization.">](https://learnopencv.com/histogram-of-oriented-gradients/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/hog-opencv-2026.07.27/Histogram-of-Oriented-Gradients-2026.07.27.zip)

This companion project computes a classic `64x128` Histogram of Oriented Gradients descriptor in Python 3 and C++17. The implementation uses stable OpenCV core and image-processing primitives for gradients, then performs cell voting and L2-Hys block normalization explicitly.

## Why the descriptor is implemented directly

OpenCV 4 exposes the classic `HOGDescriptor` wrapper, but that wrapper is not present in the OpenCV 5.0 API. The implementation here keeps the educational algorithm runnable on both versions:

- unsigned gradient orientations from `0` to `180` degrees;
- `8x8` cells with nine orientation bins;
- bilinear interpolation between adjacent orientation bins;
- `2x2`-cell blocks with an `8x8` stride;
- L2-Hys normalization with a `0.2` clipping threshold.

For a `64x128` window, the geometry is `7 x 15` blocks, four cells per block, and nine bins per cell: `7 x 15 x 4 x 9 = 3780` values.

## Run Python

```bash
python -m pip install -r requirements.txt
python hog.py --output-dir output
```

Without `--input`, the program creates a deterministic silhouette. With an input image, it resizes the image to the standard window. It writes the normalized input, a cell-orientation visualization, and the descriptor array.

## Build and run C++17

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
./build/hog_demo --output-dir=output
ctest --test-dir build --output-on-failure
```

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

The tests lock the descriptor length to `3780`, confirm deterministic output, verify the zero-degree bin with a horizontal gradient, verify zero gradients for a flat image, exercise input errors, and check every generated artifact. The project passed with Python OpenCV 4.13 and exact OpenCV 5.0.0, plus exact native C++ OpenCV 4.13.0 and exact 5.0.0.

The direct implementation is designed for teaching and version stability. It does not claim byte-for-byte equivalence with every historical `HOGDescriptor` build, whose color-channel selection, gamma correction, spatial interpolation, and descriptor ordering may differ.

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

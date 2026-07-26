# [Homography Examples Using OpenCV in Python and C++](https://learnopencv.com/homography-examples-using-opencv-python-c/)

[<img src="https://learnopencv.com/wp-content/uploads/2016/01/homography-example-768x511.jpg" alt="Perspective correction and virtual billboard homography examples" width="768">](https://learnopencv.com/homography-examples-using-opencv-python-c/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download code" width="200">](https://github.com/spmallick/learnopencv/releases/download/homography-v2026.07.26/Homography-2026.07.26.zip)

This directory contains tested examples for perspective correction, image-to-image mapping, and a virtual billboard. Every program supports a headless path that writes its result to disk. The interactive examples still allow four mouse clicks when `--points` is omitted.

## Tested versions

- Python 3.14, OpenCV Python 4.13, and NumPy 2.4
- C++17, CMake 3.29, and OpenCV C++ 4.12

OpenCV 4.10 through 5.x is supported.

The C++ build selects OpenCV 4's `calib3d` module or OpenCV 5's `geometry` module after detecting the installed major version. Both the local OpenCV 4 path and the official-tag OpenCV 5.0 Python and C++ test matrix are verified.

## Point order

Pass quadrilaterals as four semicolon-separated `x,y` pairs:

```text
top-left;top-right;bottom-right;bottom-left
```

For example:

```text
318,256;534,372;316,670;73,473
```

Points must form a convex quadrilateral and lie within the corresponding image. When `--points` is omitted, click the same clockwise order, press `R` to restart, and press Enter after all four points are selected.

## Python examples

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Correct the perspective of the included book without opening a window:

```bash
python perspective-correction.py \
  --input book1.jpg \
  --points "318,256;534,372;316,670;73,473" \
  --width 300 \
  --height 400 \
  --output perspective-corrected.jpg
```

Estimate a homography from the two supplied sets of book corners:

```bash
python homography_book.py --output warped-book.jpg
```

Create a virtual billboard with explicit points:

```bash
python virtual-billboard.py \
  --source first-image.jpg \
  --destination times-square.jpg \
  --points "80,165;320,255;255,390;35,305" \
  --output virtual-billboard.jpg
```

Add `--display` to any command to show its inputs and result.

`homography.py` is the compact perspective-correction example, while `homography2.py` is the generic image-compositing example retained for compatibility with the original tutorial.

## C++ examples

```bash
cmake -S . -B build
cmake --build build
```

The build creates:

- `perspective_correction`: perspective correction with optional mouse selection
- `homography`: the compact perspective-correction example
- `virtual_billboard`: billboard replacement with optional mouse selection
- `homography2`: generic image-to-quadrilateral compositing

For example:

```bash
./build/perspective_correction \
  --input book1.jpg \
  --points "318,256;534,372;316,670;73,473" \
  --output perspective-corrected-cpp.jpg
```

## Expected outputs

- Perspective correction: a `300x400` front-facing crop with the default dimensions.
- Book mapping: a `600x800` warped image aligned to `book1.jpg`.
- Virtual billboard: a `1280x854` composite matching `times-square.jpg`.

The compositing examples use a warped mask instead of adding two images. This prevents background pixels from brightening or overflowing.

## Automated tests

```bash
python -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
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

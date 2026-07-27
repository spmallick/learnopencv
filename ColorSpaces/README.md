# [Color Spaces in OpenCV with Python and C++](https://learnopencv.com/color-spaces-in-opencv-cpp-python/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2017/05/04104101/color-change-illumination.gif" alt="Rubik's Cube colors under changing illumination" width="640">](https://learnopencv.com/color-spaces-in-opencv-cpp-python/)

This companion project explores BGR, HSV, YCrCb, and Lab encodings and uses
them for color segmentation under changing illumination. It includes:

- a pixel inspector in Python and C++;
- a headless or visual segmentation program in Python and C++;
- a vectorized density-plot analysis for the bundled Rubik's Cube samples;
- deterministic Python tests and CTest validation.

OpenCV reads ordinary color images in **BGR channel order**. For 8-bit HSV,
hue is encoded from 0 through 179; the segmentation helpers support wrapped hue
ranges such as 170-179 plus 0-10 for red.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/color-spaces-opencv-2026.07.26/ColorSpaces-2026.07.26.zip)

## Requirements

- Python 3.10 or newer
- NumPy 1.23 or newer
- OpenCV 4.x or OpenCV 5.x
- Matplotlib 3.7 or newer for the density plots
- CMake 3.16 or newer and a C++17 compiler for C++

```bash
python3 -m pip install -r requirements.txt
```

## Python examples

Inspect the center pixel and save an annotated image without opening a window:

```bash
python3 interactiveColorDetect.py \
  --no-display \
  --output outputs/color-values.png
```

Segment yellow cube pieces in HSV while keeping the visualization in BGR:

```bash
python3 interactiveColorSegment.py \
  --space HSV \
  --lower 20 80 40 \
  --upper 45 255 255 \
  --no-display \
  --output-dir outputs
```

Generate a density comparison for all yellow samples:

```bash
python3 dataAnalysis.py \
  --color yellow \
  --zoom \
  --output outputs/yellow-density.png
```

Omit `--no-display` from the first two commands to use their desktop windows.
Both programs accept `--input /path/to/image`; bundled defaults are resolved
from the script directory, not the caller's working directory.

## C++ build and examples

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

./build/interactiveColorDetect \
  --no-display \
  --output outputs/color-values-cpp.png

./build/interactiveColorSegment \
  --space HSV \
  --lower 20 80 40 \
  --upper 45 255 255 \
  --no-display \
  --output-dir outputs
```

Use `-DOpenCV_DIR=/path/to/lib/cmake/opencv4` or the corresponding OpenCV 5
configuration directory to select a specific installation.

## Validation and tests

The stable one-pixel conversion regression is:

| Encoding | Values |
|---|---|
| BGR | `[40, 158, 16]` |
| HSV | `[65, 229, 158]` |
| YCrCb | `[102, 67, 93]` |
| Lab | `[145, 71, 177]` |

The test suite also checks all 10 cube images and 56 cropped piece images,
HSV hue wrapping, headless output files, malformed inputs, and execution from
an unrelated working directory.

```bash
python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
```

The direct validation commands are:

```bash
python3 interactiveColorDetect.py --validate --no-display
python3 interactiveColorSegment.py --validate --no-display
python3 dataAnalysis.py --validate
./build/interactiveColorDetect --validate --no-display
./build/interactiveColorSegment --validate --no-display
```

## Project layout

```text
ColorSpaces/
├── CMakeLists.txt
├── README.md
├── color_spaces.hpp
├── color_spaces.py
├── dataAnalysis.py
├── interactiveColorDetect.cpp
├── interactiveColorDetect.py
├── interactiveColorSegment.cpp
├── interactiveColorSegment.py
├── requirements.txt
├── images/
│   └── rub00.jpg ... rub09.jpg
├── pieces/
│   └── 56 cropped color samples
└── tests/
    └── test_color_spaces.py
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

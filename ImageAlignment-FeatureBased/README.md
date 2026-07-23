# Feature Based Image Alignment using OpenCV

This repository contains code for the blog post [Feature Based Image Alignment using OpenCV (C++/Python)](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/).

<img src="https://learnopencv.com/wp-content/uploads/2018/03/image-alignment-using-opencv.jpg" alt="Image Alignment" width="900">

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/7actm7m3yu4rg01mhve9b/h?dl=1&rlkey=7irdzdwlwniboxv6n0nyh4puh)

## OpenCV compatibility

The Python and C++ examples are tested with exact OpenCV 4.14.0 and OpenCV
5.0.0 releases. They require Python 3.9 or newer for Python, and CMake 3.16 or
newer plus C++17 for C++. ORB is part of the main OpenCV distribution, so
`opencv_contrib` is not required.

OpenCV 5 renamed the C++ `features2d` module to `features` and moved
`findHomography` from `calib3d` to `geometry`. The source selects the appropriate
headers and CMake components for each supported major version. The Python API
names are unchanged.

## Python

From this directory, create an environment and install the supported dependency
ranges:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the bundled example headlessly and validate the result:

```bash
python align.py --output-dir output --no-display --validate
```

The default input images are resolved relative to `align.py`, so the command can
also be run from another working directory by using the script's absolute path.
`--output-dir` is resolved from the caller's working directory.

Use different images with:

```bash
python align.py \
  --input /path/to/image.jpg \
  --reference /path/to/reference.jpg \
  --output-dir output \
  --no-display \
  --validate
```

Run all Python regressions:

```bash
python -m unittest discover -s tests -v
```

## C++

Configure with the OpenCV installation to test. Standard Unix installations
typically place `OpenCVConfig.cmake` under `lib/cmake/opencv4` for OpenCV 4 and
`lib/cmake/opencv5` for OpenCV 5.

```bash
cmake -S . -B build \
  -DOpenCV_DIR=/path/to/opencv-5.0.0/lib/cmake/opencv5 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Run and test the executable:

```bash
./build/image_alignment --output-dir output --no-display --validate
ctest --test-dir build --output-on-failure
```

As in the Python example, CMake embeds the project source directory for the
default bundled inputs, and explicit `--input` and `--reference` paths override
them.

## Command-line options

Both implementations accept the same options:

| Option | Meaning |
| --- | --- |
| `--input PATH` | Image that will be aligned; defaults to `scanned-form.jpg`. |
| `--reference PATH` | Reference image; defaults to `form.jpg`. |
| `--output-dir PATH` | Directory for `aligned.jpg` and `matches.jpg`; defaults to the current directory. |
| `--no-display` | Explicitly select the headless workflow. The example does not open GUI windows. |
| `--validate` | Check output dimensions, readability, and alignment quality. |

## Regression contract

For the bundled images, ORB finds 500 keypoints in each image and 500 raw
matches. The nominal best 15% ends at a tied Hamming distance, so the
implementations retain all matches at that distance: 78 matches with 51 RANSAC
inliers. This avoids selecting an arbitrary tied match based on collection
ordering.

Validation requires:

- `aligned.jpg` to be 1000 by 1293 pixels and `matches.jpg` to be 2000 by 1333
  pixels;
- both files to be readable and nonempty;
- the aligned image's mean absolute error against the reference to be at least
  60% lower than the resized unaligned input; and
- the explicit `Alignment validation passed` marker.

OpenCV 5 changed homography refinement and perspective-warp interpolation.
Small homography and pixel differences between 4.14 and 5.0 are therefore
expected; the tests compare stable counts, dimensions, inliers, and alignment
quality instead of encoded JPEG bytes.

## Project layout

```text
ImageAlignment-FeatureBased/
├── CMakeLists.txt
├── README.md
├── align.cpp
├── align.py
├── form.jpg
├── requirements.txt
├── scanned-form.jpg
└── tests/
    ├── check_cpp_cli.cmake
    └── test_alignment.py
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

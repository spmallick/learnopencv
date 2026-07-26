# [Augmented Reality Using ArUco Markers in OpenCV](https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/)

This directory contains the tested C++ and Python companion code for the
[LearnOpenCV article](https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/).
The original example was written by Sunita Nayak at BigVision LLC; this
refresh modernizes it for current OpenCV APIs and reproducible headless tests.

<a href="https://learnopencv.com/augmented-reality-using-aruco-markers-in-opencv-c-python/"><img src="https://cdn.learnopencv.com/wp-content/uploads/2020/03/04095302/augmented-reality-aruco-markers-opencv.jpg" alt="Augmented reality using ArUco markers in OpenCV" width="900"></a>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://github.com/spmallick/learnopencv/releases/download/aruco-augmented-reality-v2026.07.26/AugmentedRealityWithArucoMarkers-2026.07.26.zip)

The example detects the four `DICT_6X6_250` markers in `test.jpg`, computes the
quadrilateral bounded by marker IDs 25, 33, 30, and 23, and warps
`new_scenery.jpg` into that region. It uses the current stateful
`ArucoDetector` and `Dictionary.generateImageMarker` APIs.

If a frame is missing any required marker, the frame is preserved instead of
indexing beyond the detected marker list. Use `--strict` when missing markers
should make a reproducible test fail.

## Requirements

- Python 3.10 or newer
- OpenCV 4.8 or newer with the ArUco module
- CMake 3.16 or newer and a C++17 compiler for the C++ examples

```bash
python3 -m pip install -r requirements.txt
```

## Python

The bundled image and overlay are the defaults:

```bash
python3 augmented_reality_with_aruco.py --strict
```

Process explicit image or video inputs:

```bash
python3 augmented_reality_with_aruco.py \
  --image test.jpg \
  --overlay new_scenery.jpg \
  --output test_ar_out_py.jpg \
  --strict

python3 augmented_reality_with_aruco.py \
  --video test.mp4 \
  --overlay new_scenery.jpg \
  --output test_ar_out_py.avi
```

Outputs contain the original and augmented frames side by side. Pass
`--augmented-only` to write only the augmented frame, and `--display` to open a
GUI window. A webcam can be selected with `--camera 0`.

Generate a printable marker with a white margin around the saved image when
placing it in a scene:

```bash
python3 generate_aruco_markers.py \
  --id 33 \
  --size 200 \
  --output marker33.png
```

## C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --config Release

./build/AugmentedRealityWithAruco \
  --image=test.jpg \
  --overlay=new_scenery.jpg \
  --output=test_ar_out_cpp.jpg \
  --strict=true

./build/GenerateArucoMarker \
  --id=33 \
  --size=200 \
  --output=marker33-cpp.png
```

## Tests

```bash
python3 -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
```

The Python suite verifies detection and augmentation of the bundled sample,
safe behavior when markers are missing, marker generation/detection round-trip,
and headless image output. CTest exercises both compiled programs.

The compatibility matrix passed Python 4/4 and CTest 2/2 against the official
OpenCV 5.0.0 tag. Local OpenCV 4 validation passed Python 4/4 with 4.13.0 and
CTest 2/2 with native 4.12.0.

OpenCV 5 moves `getPerspectiveTransform` to the `geometry` component. The
CMake project selects that component and its `geometry/2d.hpp` header only for
OpenCV 5, while OpenCV 4 continues to use `imgproc`.

The API choices follow OpenCV's current
[ArUco marker detection tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html).



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

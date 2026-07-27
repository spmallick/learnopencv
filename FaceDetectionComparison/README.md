# [Face Detection Comparison with OpenCV, YuNet, Haar, and dlib](https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/)

<p align="center">
  <a href="https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/">
    <img src="https://cdn.learnopencv.com/wp-content/uploads/2018/10/04095723/Face-Detection.jpg" alt="Face detection comparison with OpenCV and dlib" width="800">
  </a>
</p>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/face-detection-comparison-opencv-2026.07.26/FaceDetectionComparison-2026.07.26.zip)

The primary Python and C++ examples now use OpenCV `FaceDetectorYN` with the
official dynamic-input YuNet model. They accept images and videos, save
correctly sized output by default, and run headlessly unless `--display` is
requested.

The filenames `face_detection_opencv_dnn.*` are retained for continuity with
the original article, but the current implementation no longer loads Caffe or
TensorFlow SSD files. Haar and dlib HOG remain explicit, optional historical
baselines. MMOD and the original SSD assets are legacy material only.

## Compatibility

- Compatibility target: OpenCV 4.14.0 and OpenCV 5.0.0
- Python: 3.10 or newer
- CMake: 3.16 or newer
- C++: C++17 compiler
- Primary dependency: OpenCV and the downloaded YuNet ONNX model
- Optional Python dependency: dlib for the HOG baseline
- Optional C++ dependency: a discoverable `dlib::dlib` CMake package

The July 2026 migration was exercised locally with Python OpenCV 4.13.0,
C++ OpenCV 4.12.0, and both languages against official OpenCV 5.0.0. Exact
OpenCV 4.14.0 remains the release acceptance target.

OpenCV 5 still provides `FaceDetectorYN`, but the historical
`CascadeClassifier` path is no longer part of its core objdetect API. CMake
therefore disables the optional Haar C++ executable on OpenCV 5. Python reports
a clear error if the installed binding has no `CascadeClassifier`.

## Setup

From this directory:

```shell
python3 -m pip install -r requirements.txt
python3 download_models.py
```

The downloader retrieves `face_detection_yunet_2026may.onnx` from OpenCV Zoo
commit `47534e27c9851bb1128ccc0102f1145e27f23f98`. It requires exactly
229,738 bytes and SHA-256
`ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`.
See [MODEL_LICENSES.md](MODEL_LICENSES.md) for the full provenance and MIT
license record.

## Primary Python example

The bundled default is `videos/baby.mp4`; paths for all bundled defaults are
resolved relative to the script rather than the caller's current directory.

```shell
# Video: writes output/yunet-video.avi.
python3 face_detection_opencv_dnn.py \
  --max-frames 2 \
  --no-display \
  --validate

# Image: writes output/yunet-image.jpg.
python3 face_detection_opencv_dnn.py \
  --input /absolute/path/to/image.jpg \
  --mode image \
  --no-display \
  --validate
```

Options include `--input`, `--mode auto|image|video`, `--model`,
`--output-dir`, `--device cpu|cuda`, `--score-threshold`,
`--nms-threshold`, `--top-k`, `--max-frames`, `--display`,
`--no-display`, and `--validate`.

## Primary C++ example

Download the model before configuring with tests enabled:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run the current YuNet path:

```shell
./build/face_detection_opencv_dnn \
  --max-frames 2 \
  --no-display \
  --validate

./build/face_detection_opencv_dnn \
  --input /absolute/path/to/image.jpg \
  --mode image \
  --no-display \
  --validate
```

Select an exact OpenCV installation with
`-DOpenCV_DIR=/absolute/path/to/lib/cmake/opencv5`. Configure with
`-DBUILD_TESTING=OFF` only when intentionally compiling before downloading the
model.

## Detector comparison

`run-all.py` and `run-all` use the dependency-free YuNet path by default.
Request optional baselines explicitly:

```shell
# OpenCV 4 build/binding with Haar available.
python3 run-all.py \
  --detectors yunet,haar \
  --max-frames 2 \
  --no-display \
  --validate

./build/run-all \
  --detectors yunet,haar \
  --max-frames 2 \
  --no-display \
  --validate
```

Each detector receives the same original frame. Output panels are concatenated
horizontally, so an input of width `W` and height `H` produces
`(number of detectors × W) × H`. The video writer is created with that exact
size; the old 2W-by-2H-frame-to-W-by-H-writer bug is removed.

For the optional Python dlib HOG baseline:

```shell
python3 -m pip install -r requirements-dlib.txt
python3 face_detection_dlib_hog.py \
  --max-frames 2 \
  --no-display \
  --validate

python3 run-all.py \
  --detectors yunet,hog \
  --max-frames 2 \
  --no-display \
  --validate
```

When CMake finds `dlib::dlib`, it also builds `face_detection_dlib_hog` and
enables `hog` in the C++ comparison program. It never unpacks the historical
`dlib.zip`.

## Regression tests

The default tests require only NumPy, OpenCV, and the verified YuNet model.
dlib, Haar, Caffe, TensorFlow, and MMOD are not loaded.

```shell
python3 -m unittest discover -s python/tests -v
ctest --test-dir build --output-on-failure
```

Python tests exercise the real image, video, and comparison CLIs from unrelated
temporary directories. They verify two YuNet faces on the bundled baby's first
frame at the documented threshold, readable same-size image output, exact
720x720 two-frame video output, comparison geometry, and missing-input errors.

C++ CTest runs two-frame YuNet and YuNet-only comparison videos. All C++ targets
compile with strict warnings (`-Wall -Wextra -Wpedantic -Werror` on Clang/GCC).
Validation checks output shape, finite box and landmark values, positive box
sizes, confidence bounds, clipped image bounds, output dimensions, and frame
counts rather than codec-dependent bytes.

## Historical files

The following remain for readers comparing the 2018 material, but are excluded
from the current build, downloader, and default test path:

- `face_detection_dlib_mmod.py` and `.cpp`
- `models/mmod_human_face_detector.dat`
- `models/deploy.prototxt`
- `models/res10_300x300_ssd_iter_140000_fp16.caffemodel`
- `models/opencv_face_detector.pbtxt`
- `models/opencv_face_detector_uint8.pb`
- `dlib.zip`

## Project layout

```text
FaceDetectionComparison/
├── .gitignore
├── CMakeLists.txt
├── MODEL_LICENSES.md
├── README.md
├── dlib.zip                         # historical snapshot
├── download_models.py
├── face_detection.hpp
├── face_detection.py
├── face_detection_dlib_hog.cpp      # optional
├── face_detection_dlib_hog.py       # optional
├── face_detection_dlib_mmod.cpp     # legacy
├── face_detection_dlib_mmod.py      # legacy
├── face_detection_opencv_dnn.cpp    # current YuNet path
├── face_detection_opencv_dnn.py     # current YuNet path
├── face_detection_opencv_haar.cpp   # optional OpenCV 4 path
├── face_detection_opencv_haar.py    # optional
├── historical_detector_cli.py
├── models/
│   ├── .gitignore
│   ├── deploy.prototxt              # legacy
│   ├── haarcascade_frontalface_default.xml
│   ├── mmod_human_face_detector.dat # legacy
│   ├── opencv_face_detector.pbtxt   # legacy
│   ├── opencv_face_detector_uint8.pb # legacy
│   └── res10_300x300_ssd_iter_140000_fp16.caffemodel # legacy
├── python/
│   └── tests/
│       └── test_face_cli.py
├── requirements-dlib.txt
├── requirements.txt
├── run-all.cpp
├── run-all.py
└── videos/
    ├── baby.mp4
    ├── neck-exercise.mp4
    └── rowing.mp4
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

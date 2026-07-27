# [Human Pose Estimation with OpenCV and MediaPipe Pose](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)

<p align="center">
  <a href="https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/">
    <img src="https://cdn.learnopencv.com/wp-content/uploads/2018/05/04100015/opencv-openpose.jpg" alt="Human pose estimation with OpenCV" width="800">
  </a>
</p>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/human-pose-opencv-2026.07.26/OpenPose-2026.07.26.zip)

This companion project estimates the landmarks of one person in an image or
video with the official OpenCV Zoo MediaPipe Pose ONNX model. The current
Python and C++ paths save their results by default and run headlessly unless
`--display` is requested.

The runnable examples no longer load Caffe. OpenCV 5 removed its Caffe parser,
and the original CMU OpenPose weights have different licensing constraints.
See [MODEL_LICENSES.md](MODEL_LICENSES.md) for the exact model provenance,
checksum, and legacy boundary.

## Compatibility

- Compatibility target: OpenCV 4.14.0 and OpenCV 5.0.0
- Python: 3.10 or newer
- CMake: 3.16 or newer
- C++: C++17 compiler
- Default backend and target: OpenCV's compatible DNN graph engine on CPU
- Optional device: CUDA, when OpenCV was built with CUDA DNN support

The July 2026 migration was exercised locally with Python OpenCV 4.13.0,
C++ OpenCV 4.12.0, and both Python and C++ against official OpenCV 5.0.0.
Exact OpenCV 4.14.0 remains the release acceptance target.

## Setup

From this directory, install the Python dependencies and download the pinned
model:

```shell
python3 -m pip install -r requirements.txt
python3 download_models.py
```

`download_models.py` downloads exactly 5,557,238 bytes from OpenCV Zoo commit
`47534e27c9851bb1128ccc0102f1145e27f23f98` and requires SHA-256
`9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`.
The legacy `./getModels.sh` command delegates to the same verifier.

## Python examples

The defaults are resolved relative to the script, so these commands also work
when invoked from another current working directory.

```shell
# Image: writes output/pose-image.jpg.
python3 OpenPoseImage.py --no-display --validate

# Video: writes output/pose-video.avi without discarding the first frame.
python3 OpenPoseVideo.py --no-display --validate

# A short deterministic smoke run.
python3 OpenPoseVideo.py \
  --input sample_video.mp4 \
  --output-dir output/smoke \
  --max-frames 2 \
  --no-display \
  --validate
```

Both programs accept `--input`, `--model`, `--output-dir`, `--device`,
`--score-threshold`, `--display`, `--no-display`, and `--validate`. The video
program also accepts `--max-frames`; zero processes the complete video.

## C++ examples

Download the model before configuring with tests enabled:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Run the image and video programs:

```shell
./build/OpenPoseImage --no-display --validate
./build/OpenPoseVideo --max-frames 2 --no-display --validate
```

Select a non-default OpenCV installation with
`-DOpenCV_DIR=/absolute/path/to/lib/cmake/opencv5`. Configure with
`-DBUILD_TESTING=OFF` only when you intentionally want to compile before
downloading the model.

## Regression tests

The Python tests invoke the real CLIs from temporary, unrelated directories.
They check validation markers, same-size readable image output, exact video
dimensions and requested frame count, and clear missing-input errors.

```shell
python3 -m unittest discover -s python/tests -v
```

The two CTest regressions run the C++ image example and a two-frame video
example with strict compiler warnings (`-Wall -Wextra -Wpedantic -Werror` on
Clang/GCC).

Validation intentionally checks stable structural behavior—33 finite
landmarks, confidence bounds, skeleton bounds, output readability, dimensions,
and frame counts—rather than JPEG bytes or exact floating-point coordinates
that may vary slightly by platform or DNN engine.

## Single-person scope

MediaPipe Pose is a landmark estimator, not a multi-person detector. These
examples letterbox the complete frame as one person's region of interest. For
best results, use an upright subject whose full body occupies most of the
frame. A production multi-person system should detect people first and pass
one normalized crop at a time.

## Legacy material

`OpenPose_Notebook.ipynb` and `pose/` are retained only as historical material
from the 2018 CMU OpenPose tutorial. They are not imported, compiled,
downloaded, or tested by the current pipeline. No CMU Caffe weights are
included or fetched.

## Project layout

```text
OpenPose/
├── .gitignore
├── CMakeLists.txt
├── MODEL_LICENSES.md
├── OpenPoseImage.cpp
├── OpenPoseImage.py
├── OpenPoseVideo.cpp
├── OpenPoseVideo.py
├── OpenPose_Notebook.ipynb       # historical Caffe notebook
├── README.md
├── download_models.py
├── getModels.sh
├── models/
│   └── .gitkeep                  # downloaded ONNX is ignored
├── multiple.jpeg
├── pose/                         # historical Caffe prototxt files
│   ├── coco/
│   │   └── pose_deploy_linevec.prototxt
│   └── mpi/
│       ├── pose_deploy_linevec.prototxt
│       └── pose_deploy_linevec_faster_4_stages.prototxt
├── pose_estimation.hpp
├── pose_estimation.py
├── python/
│   └── tests/
│       └── test_pose_cli.py
├── requirements.txt
├── sample_video.mp4
└── single.jpeg
```

## Model license

The current MediaPipe Pose model folder in the pinned OpenCV Zoo revision is
licensed under Apache License 2.0. The downloader records and verifies the Git
LFS object digest; this repository does not redistribute the ONNX file.

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

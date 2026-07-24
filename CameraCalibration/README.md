# Camera Calibration Using OpenCV

**This repository contains the code for [Camera Calibration Using Opencv](https://learnopencv.com/camera-calibration-using-opencv/) blog post**.

Download the immutable, versioned
[CameraCalibration.zip](https://github.com/spmallick/learnopencv/releases/download/camera-calibration-opencv-2026.07.23/CameraCalibration.zip)
bundle and its
[SHA-256 checksum](https://github.com/spmallick/learnopencv/releases/download/camera-calibration-opencv-2026.07.23/CameraCalibration.zip.sha256).
The ZIP contains exactly one top-level `CameraCalibration/` directory with the
files shown in the directory layout below.

Calibrate a camera from checkerboard images with OpenCV 4 or OpenCV 5. The
Python and C++ implementations share the same algorithm and use all 41 bundled
images. Both report OpenCV's calibration RMS and an independently calculated
reprojection RMSE.

## Requirements

- Python 3.10 or newer
- NumPy and OpenCV from `requirements.txt`
- CMake 3.16 or newer and a C++17 compiler for C++

The compatibility matrix was validated with Python 3.14, NumPy 2.4.2, and the
exact OpenCV 4.14.0 and OpenCV 5.0.0 releases. The dependency ranges permit
other OpenCV 4 and 5 releases, but those were not part of this regression
baseline.

Run all commands below from this project folder:

```shell
cd CameraCalibration
```

Install the Python dependencies in a virtual environment:

```shell
python3 -m pip install -r requirements.txt
```

## Python

Run calibration with the bundled images:

```shell
python3 cameraCalibration.py
```

Run calibration followed by both undistortion methods:

```shell
python3 cameraCalibrationWithUndistortion.py
```

The interactive mode pauses after each matched image so every successful or
failed checkerboard detection can be inspected. For a headless
run with numerical validation:

```shell
python3 cameraCalibration.py --no-display --validate
python3 cameraCalibrationWithUndistortion.py \
  --no-display \
  --validate \
  --output-dir output
```

Both scripts resolve the bundled images relative to their own location, so
these headless commands also work when invoked through absolute paths from an
unrelated current directory.

## C++

Configure, build, and test both executables:

```shell
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

For OpenCV 5, point `OpenCV_DIR` at that installation's `OpenCVConfig.cmake`
directory instead:

```shell
cmake -S . -B build-opencv5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5
cmake --build build-opencv5 --config Release
ctest --test-dir build-opencv5 -C Release --output-on-failure
```

An OpenCV 5 build must include the `calib`, `geometry`, and `objdetect`
modules. They replace the OpenCV 4 `calib3d` dependency used by this tutorial.
All tutorial targets compile with `-Wall -Wextra -Wpedantic -Werror` on
GCC/Clang or `/W4 /WX` on MSVC.

Run the executables:

```shell
./build/cameraCalibration
./build/cameraCalibrationWithUndistortion
```

For a headless validation run:

```shell
./build/cameraCalibration --no-display --validate
./build/cameraCalibrationWithUndistortion \
  --no-display \
  --validate \
  --output-dir output
```

With a multi-configuration generator such as Visual Studio, the executable may
be under `build/Release/`. CMake embeds the bundled image directory in both
executables, so they do not depend on the current working directory.

## Command-line options

Both languages accept the same functional options:

| Option | Programs | Behavior |
| --- | --- | --- |
| `--images GLOB` | Both | Use another calibration image pattern. Quote the glob in a shell. |
| `--no-display` | Both | Skip all OpenCV windows and keyboard waits. |
| `--validate` | Both | Check camera-matrix, pose-count, and reprojection invariants. |
| `--sample IMAGE` | Undistortion | Undistort this image instead of the first sorted calibration image. It must have the calibration resolution. |
| `--output-dir PATH` | Undistortion | Create the directory and write both result images there. The default is the current directory. |
| `-h`, `--help` | Both | Print the Python or C++ command help. |

The undistortion program always writes exactly:

- `undistorted_direct.jpg` from `undistort`
- `undistorted_remap.jpg` from `initUndistortRectifyMap` followed by `remap`

## Regression tests

Run the Python numerical regression suite with:

```shell
python3 -m unittest discover -s tests -v
```

The six Python tests exercise the shared functions and both real scripts from
an unrelated temporary directory. The five CTest tests exercise both real C++
executables, the fixture-specific C++ oracle, and explicit CLI failures. The
C++ undistortion test clears its build-local output directory first, requires
exactly the two documented filenames, rejects empty files, and has the
executable decode both JPEGs to verify their dimensions.

Both programs print the intrinsic matrix, distortion coefficients, and one
rotation and translation vector for every successfully detected checkerboard,
preserving the original tutorial's complete calibration output.

The stable bundled-fixture expectations are:

- 41 readable images and 41 complete checkerboard detections
- image size `640x480`
- calibration RMS approximately `0.2603201` pixels
- independently calculated reprojection RMSE approximately `0.2603200` pixels
- camera matrix approximately:

  ```text
  [503.5118,   0.0000, 313.4135]
  [  0.0000, 503.1461, 243.0911]
  [  0.0000,   0.0000,   1.0000]
  ```

OpenCV 4.14 returns Python checkerboard coordinates with shape `(N, 1, 2)`,
whereas OpenCV 5.0 returns `(N, 2)`. The shared Python implementation
normalizes these equivalent representations before computing error. The
direct and precomputed-map undistortion paths also have small
version-dependent interpolation differences; validation requires their mean
absolute pixel difference to remain below `0.1`, rather than requiring
byte-identical images.

## Directory structure

```text
CameraCalibration/
├── CMakeLists.txt
├── README.md
├── calibration_utils.hpp
├── calibration_utils.py
├── cameraCalibration.cpp
├── cameraCalibration.py
├── cameraCalibrationWithUndistortion.cpp
├── cameraCalibrationWithUndistortion.py
├── images/
│   ├── image_10.jpg
│   ├── image_11.jpg
│   ├── image_12.jpg
│   ├── image_14.jpg
│   ├── image_15.jpg
│   ├── image_18.jpg
│   ├── image_19.jpg
│   ├── image_2.jpg
│   ├── image_20.jpg
│   ├── image_21.jpg
│   ├── image_22.jpg
│   ├── image_23.jpg
│   ├── image_24.jpg
│   ├── image_25.jpg
│   ├── image_26.jpg
│   ├── image_27.jpg
│   ├── image_29.jpg
│   ├── image_30.jpg
│   ├── image_40.jpg
│   ├── image_41.jpg
│   ├── image_42.jpg
│   ├── image_43.jpg
│   ├── image_44.jpg
│   ├── image_46.jpg
│   ├── image_47.jpg
│   ├── image_48.jpg
│   ├── image_49.jpg
│   ├── image_5.jpg
│   ├── image_50.jpg
│   ├── image_57.jpg
│   ├── image_6.jpg
│   ├── image_62.jpg
│   ├── image_64.jpg
│   ├── image_65.jpg
│   ├── image_70.jpg
│   ├── image_71.jpg
│   ├── image_72.jpg
│   ├── image_73.jpg
│   ├── image_75.jpg
│   ├── image_8.jpg
│   └── image_9.jpg
├── requirements.txt
└── tests/
    ├── assert_cli_failure.cmake
    ├── run_undistortion_cli.cmake
    ├── test_camera_calibration.cpp
    └── test_camera_calibration.py
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

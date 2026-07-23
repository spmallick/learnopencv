# Find frame rate (frames per second/FPS) in OpenCV

This directory contains the Python and C++ examples for [Find frame rate (frames per second/FPS) in OpenCV](https://learnopencv.com/how-to-find-frame-rate-or-frames-per-second-fps-in-opencv-python-cpp/).

The compatibility target is:

- OpenCV 4.14.0 and OpenCV 5.0.0
- Python 3.9 or newer
- CMake 3.16 or newer
- A C++17 compiler

The examples use APIs shared by OpenCV 4.x and 5.x, and CMake rejects major versions other than 4 and 5. The automated acceptance matrix is run against the exact OpenCV 4.14.0 and 5.0.0 releases.

## Download the standalone project

Download the immutable, versioned [FPS.zip](https://github.com/spmallick/learnopencv/releases/download/fps-opencv-2026.07.23/FPS.zip) bundle and its [SHA-256 checksum](https://github.com/spmallick/learnopencv/releases/download/fps-opencv-2026.07.23/FPS.zip.sha256). The ZIP contains exactly one top-level `FPS/` directory with the eight files shown in the directory layout below.

On macOS or Linux, download and verify both files before extracting the project:

```bash
curl --fail --location --remote-name \
  https://github.com/spmallick/learnopencv/releases/download/fps-opencv-2026.07.23/FPS.zip
curl --fail --location --remote-name \
  https://github.com/spmallick/learnopencv/releases/download/fps-opencv-2026.07.23/FPS.zip.sha256
shasum -a 256 -c FPS.zip.sha256
unzip FPS.zip
cd FPS
```

The expected SHA-256 digest for `FPS.zip` is `12e1e6fd48c4fdc00907c970d73616d1cbc1ef85fb9a353843299ebbab311bbf`.

## What the example measures

The examples report two different values:

1. `CAP_PROP_FPS` asks the active video backend for its reported frame-rate property. The value is backend-, driver-, and device-dependent; it is not necessarily declared metadata. Unsupported or invalid values are normalized to zero: OpenCV 4 commonly returns `0`, while OpenCV 5 introduced `-1` for an unavailable property. This behavior follows the official [OpenCV 4-to-5 Video I/O migration guidance](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration#7-video-io-module).
2. The timed frame-read rate measures a bounded loop of `VideoCapture.read()` attempts with a monotonic clock, then divides the successful frame count by the total elapsed time. Because `read()` grabs and decodes a frame, the result is delivered frame-read throughput for a camera and read/decode throughput for a file. Backend buffering and latency mean it is not guaranteed to be a camera sensor's acquisition rate. It is also not the video's playback FPS and should not be compared with `CAP_PROP_FPS` as a performance claim.

Warm-up reads happen before timing starts. If a short file ends before the requested maximum, the numerator uses the actual number of successfully decoded frames; the elapsed interval includes the final failed read that detects end-of-file.

Neither example displays frames or writes output files, so `--no-display` and `--output-dir` options are not applicable.

## Install the Python dependencies

Run the documented commands from this `FPS` directory:

```bash
cd /path/to/learnopencv/FPS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in dependency ranges admit both supported OpenCV major versions. Exact OpenCV 4.14.0 and 5.0.0 bindings should be selected in isolated validation environments for the acceptance matrix.

## Python

Measure camera 0 using the original 120-frame default:

```bash
python frame_rate.py
```

Measure a video file, discarding ten warm-up frames before timing:

```bash
python frame_rate.py \
  --source /absolute/path/to/video.mp4 \
  --frames 300 \
  --warmup-frames 10 \
  --validate
```

`--validate` checks machine-independent measurement invariants and prints `Validation passed` only after they succeed.

## C++

Configure a strict Release build and select the Python interpreter used by CTest. The commands below assume a single-config Unix-like generator and a C++ OpenCV installation matching the interpreter's `cv2` version:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build --config Release
```

Run the same camera or file measurements:

```bash
./build/frame_rate
./build/frame_rate \
  --source /absolute/path/to/video.mp4 \
  --frames 300 \
  --warmup-frames 10 \
  --validate
```

With a multi-config generator such as Visual Studio, the Release executable is normally under `build/Release/` and may have an `.exe` suffix.

To choose an exact OpenCV installation, set `OpenCV_DIR` to the directory containing its `OpenCVConfig.cmake`. Installation layouts vary, but common examples are:

```bash
cmake -S . -B build-opencv-4.14 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv-4.14.0/lib/cmake/opencv4 \
  -DPython3_EXECUTABLE=/path/to/opencv-4.14-python

cmake -S . -B build-opencv-5.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv-5.0.0/lib/cmake/opencv5 \
  -DPython3_EXECUTABLE=/path/to/opencv-5.0-python
```

The configured Python interpreter must import the same exact OpenCV version selected by `OpenCV_DIR`; the cross-language test checks this. Configure with `-DBUILD_TESTING=OFF` when only the C++ executable is required and Python test dependencies are intentionally unavailable.

## Command-line options

Both implementations accept the same controls:

- `--source VALUE`: camera index, video path, URL, or image-sequence pattern; default `0`
- `--frames N`: maximum number of frames to time; default `120`
- `--warmup-frames N`: successful reads to discard before timing; default `0`
- `--validate`: check intrinsic invariants and print the success marker
- `--help`: print usage without opening the default camera

A signed integer within OpenCV's C++ `int` range selects a camera index. Oversized integers remain string sources. Qualify an in-range numeric filename with a path component, such as `./123`, to make it a file source. Relative paths are resolved from the caller's current working directory; there are no bundled runtime inputs.

## Tests

Run the Python unit and real-CLI suite from `FPS`:

```bash
python -m unittest discover -s tests -v
```

Exact matrix runs set the expected version explicitly so an environment mistake fails before semantic testing:

```bash
FPS_EXPECTED_OPENCV_VERSION=4.14.0 \
  python -m unittest discover -s tests -v

FPS_EXPECTED_OPENCV_VERSION=5.0.0 \
  python -m unittest discover -s tests -v
```

The 16 tests cover source parsing and camera-index bounds, the OpenCV 4 `0` and OpenCV 5 `-1` unavailable-property values, non-finite backend values and clock values, warm-up behavior, early end-of-file, validation, invalid CLI input, missing and oversized sources, and the real Python entry point from an unrelated working directory. Every run also logs its active `cv2.__version__`.

After a CMake build, run the C++ and cross-language regression:

```bash
ctest --test-dir build --build-config Release --output-on-failure
```

CTest creates and preflights a temporary 12-frame, 64-by-48 MJPG/AVI file with 15 FPS metadata. It runs the real Python and C++ entry points on that same absolute source from an unrelated working directory and checks three successful modes:

- Three warm-up reads followed by a request for 20 timed frames produces exactly 9 timed frames at end-of-file.
- Two warm-up reads followed by a request for 4 timed frames produces exactly 4 timed frames.
- A normal request without `--validate` succeeds and does not print the validation marker.

The regression also checks help, argument errors, missing option values, warm-up exhaustion, missing and oversized sources, aligned exit codes, and the absence of uncaught exceptions. It requires matching exact OpenCV runtime versions, equal frame counts, `15.000` backend-reported FPS within a `0.05` tolerance, finite positive timing values, correct validation-marker behavior, and no output artifacts. It deliberately does not compare elapsed time or timed throughput across languages, versions, or machines.

The completed acceptance matrix is:

| Language and check | OpenCV 4.14.0 | OpenCV 5.0.0 |
| --- | --- | --- |
| Python unit and real-CLI tests | 16 passed | 16 passed |
| C++17 strict Release build | Passed | Passed |
| Cross-language CTest | 1 passed | 1 passed |

## Directory layout

```text
FPS/
├── CMakeLists.txt
├── README.md
├── frame_rate.cpp
├── frame_rate.py
├── requirements.txt
└── tests/
    ├── compare_implementations.py
    ├── test_frame_rate.py
    └── video_fixture.py
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

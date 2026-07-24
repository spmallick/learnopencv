# Video Stabilization Using Point Feature Matching in OpenCV

This folder contains the Python and C++ examples for the
[Video Stabilization Using Point Feature Matching in OpenCV](https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/)
tutorial. Both examples track point features between adjacent frames, estimate
partial-affine camera motion, smooth the resulting trajectory, and write the
original and stabilized views side by side.

## Compatibility contract

- OpenCV 4.14.0
- OpenCV 5.0.0
- Python 3.9 or newer
- NumPy 1.23 or newer
- CMake 3.16 or newer
- A C++17 compiler

OpenCV 4.14 is still identified as `4.14.0-pre` in the upstream development
documentation at the time of this update. The code targets its current API and
will be retested against the final `4.14.0` packages when they are published.
OpenCV 5 validation uses the released `5.0.0` packages.

The checked-in dependency ranges admit both supported OpenCV majors:

```text
numpy>=1.23,<3
opencv-python>=4.8,<6
```

The dependency range intentionally remains broad enough to install on systems
where the final OpenCV 4.14 Python wheel is not published yet. The examples use
`estimateAffinePartial2D` and the modern two-value Python return contract, which
are shared by OpenCV 4.14 and 5.0. The removed OpenCV 3-only
`estimateRigidTransform` compatibility branch is no longer needed.

## Input and output

The bundled `video.mp4` is resolved relative to the source file, so the default
commands work from the project directory or from an unrelated current
directory. Use `--input PATH` to process another video.

Output defaults to `output/video_out.mp4`. The output directory is created
automatically. Use `--output-dir PATH` and `--output-name NAME` to change it.
The video contains the original frame on the left and stabilized frame on the
right. For wide input, the comparison is resized to at most 1920 pixels wide.

Useful options shared by both implementations:

- `--input PATH`: choose an input video.
- `--output-dir PATH`: choose the output directory.
- `--output-name NAME`: choose the output filename.
- `--smoothing-radius N`: set the non-negative moving-average radius.
- `--no-display`: disable the preview window for headless use.
- `--validate`: check the output frame count, dimensions, and readability, then
  print `VALIDATION PASSED`.

Press Escape to stop an interactive preview early.

## Run the Python example

From `VideoStabilization`:

```shell
/Users/spmallick/.venv/codex/bin/python -m pip install -r requirements.txt
/Users/spmallick/.venv/codex/bin/python video_stabilization.py
```

For a repeatable headless run:

```shell
/Users/spmallick/.venv/codex/bin/python video_stabilization.py \
  --no-display --validate
```

The script can also be launched from any directory:

```shell
/Users/spmallick/.venv/codex/bin/python \
  /absolute/path/to/VideoStabilization/video_stabilization.py \
  --output-dir /tmp/video-stabilization --no-display --validate
```

Run the Python regression tests from the repository root:

```shell
/Users/spmallick/.venv/codex/bin/python \
  -m unittest discover -s VideoStabilization/tests -v
```

The tests exercise the real command-line entry point from an unrelated
directory, validate a generated synthetic video, check smoothing behavior, and
verify the missing-input error.

## Build and run the C++ example

From `VideoStabilization`:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/video_stabilization
```

To select a particular OpenCV installation:

```shell
cmake -S . -B build-opencv5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv-5/lib/cmake/opencv5
cmake --build build-opencv5 --parallel
```

Run the C++ regression from any directory:

```shell
/absolute/path/to/build/video_stabilization \
  --output-dir /tmp/video-stabilization-cpp \
  --no-display --validate
```

Or run the registered CTest:

```shell
ctest --test-dir build --output-on-failure
```

The build requests only the OpenCV modules used by the example, rejects
unsupported major versions, and compiles with warnings treated as errors.

## Project files

```text
VideoStabilization/
├── .gitignore
├── CMakeLists.txt
├── README.md
├── requirements.txt
├── tests/
│   └── test_video_stabilization.py
├── video.mp4
├── video_stabilization.cpp
└── video_stabilization.py
```

# Computer Vision & AI Consulting

If you need help implementing your computer vision or AI project, we provide consulting services at [BigVision.AI](https://bigvision.ai). 

Contact us at [contact@bigvision.ai](mailto:contact@bigvision.ai).

[![BigVision.AI](https://bigvision.ai/wp-content/uploads/2022/01/logo.png)](https://bigvision.ai)

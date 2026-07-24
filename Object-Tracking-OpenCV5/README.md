# OpenCV 5 Object Trackers: A Complete Guide (C++/Python)

This is the companion code for LearnOpenCV's **OpenCV 5 Object Trackers:
A Complete Guide (C++/Python)**.

One command-line application—implemented in both Python and C++—runs the six
single-object trackers in OpenCV 5's modern, non-legacy `cv::Tracker` API:

| Tracker | Type | Module | Model files |
| --- | --- | --- | --- |
| MIL | classical | main (`video`) | none |
| KCF | classical | contrib (`tracking`) | none |
| CSRT | classical | contrib (`tracking`) | none |
| DaSiamRPN | deep learning | main (`video`) | 3 ONNX files |
| NanoTrack v2 | deep learning | main (`video`) | 2 ONNX files |
| VitTrack | deep learning | main (`video`) | 1 ONNX file |

Older Boosting, MedianFlow, TLD, and MOSSE implementations remain under the
separate `cv::legacy`/`cv2.legacy` API and are outside this example's scope.
GOTURN is intentionally absent: OpenCV 5 removed its Caffe DNN importer, and
with it the GOTURN tracker. The article covers the migration path.

## Supported and tested versions

- Acceptance-tested OpenCV versions: **4.14.0** and **5.0.0**, each with
  `opencv_contrib`.
- Python 3.9+ with NumPy 1.23–2.x. The dependency range is
  `opencv-contrib-python>=4.9,<6` and `numpy>=1.23,<3`; TrackerVit establishes
  the OpenCV 4.9 API floor, while 4.14.0 and 5.0.0 are the exact validation
  targets.
- C++17 and CMake 3.16+. OpenCV's `core`, `dnn`, `imgproc`, `videoio`,
  `highgui`, and `video` modules are required. The contrib `tracking` module
  enables KCF and CSRT; without it, the other four trackers still build and the
  unavailable trackers are reported explicitly.

The full matrix was run on macOS arm64 with Python 3.14.3, CMake 3.29.3, and
AppleClang 21.0.0. Python used source-built OpenCV 4.14.0 plus contrib and the
`opencv-contrib-python` 5.0.0.93 wheel. C++ used fresh Release source builds for
both exact OpenCV versions.

## Directory structure

```text
Object-Tracking-OpenCV5/
├── .gitignore
├── README.md
├── download_models.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── cmake/
│   │   ├── RunExpectedFailure.cmake
│   │   └── RunTrackerValidation.cmake
│   └── object_tracking.cpp
└── python/
    ├── object_tracking.py
    ├── requirements.txt
    └── tests/
        ├── test_download_models.py
        └── test_object_tracking.py
```

The generated `models/` directory is deliberately absent from the tracked
tree. `download_models.py` creates it next to the script.

## Setup

Start in the project root:

```shell
cd /path/to/learnopencv/Object-Tracking-OpenCV5
```

Download the six ONNX files used by the three deep-learning trackers:

```shell
python3 download_models.py
```

The downloader uses immutable upstream revisions, bounded streaming reads, and
pinned byte sizes and SHA-256 checksums. It atomically replaces the destination
only after verification. Use `--models-dir PATH` to choose another destination
or `--force` to refresh already verified files.

### Python

```shell
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r python/requirements.txt
```

### C++

```shell
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

Point `OpenCV_DIR` at the exact installation to select a version:

```shell
cmake -S cpp -B cpp/build-5.0.0 -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/opt/opencv-5.0.0/lib/cmake/opencv5
cmake --build cpp/build-5.0.0 -j
```

## Run

Run these commands from the project root. Absolute script, executable, input,
model, and output paths also work from an unrelated current directory. A
numeric input selects a camera; any other input is treated as a video path.

```shell
# Python
python3 python/object_tracking.py --list-trackers
python3 python/object_tracking.py --tracker vittrack --input video.mp4
python3 python/object_tracking.py --tracker csrt --input 0
python3 python/object_tracking.py --tracker nanotrack --input video.mp4 \
        --bbox 300,150,120,180 --output-dir outputs --no-display

# C++
./cpp/build/object_tracking --list
./cpp/build/object_tracking --tracker=vittrack --input=video.mp4
./cpp/build/object_tracking --tracker=mil --validate --no-display \
        --output-dir=outputs
```

Both implementations support the options below. Python accepts
`--option value` or `--option=value`; C++ uses `--option=value`.

- `--tracker`: `mil`, `kcf`, `csrt`, `dasiamrpn`, `nanotrack`, or `vittrack`.
- `--input`: a video path or numeric camera index.
- `--bbox x,y,w,h`: an initial rectangle; omit it for interactive selection.
- `--models-dir PATH`: override the source-relative `models/` directory.
- `--output-dir PATH`: write `tracked_<name>.avi` and
  `metrics_<name>.json`.
- `--max-frames N`: stop after at most `N` processed frames; zero means
  continue until end-of-stream or an interactive stop.
- `--no-display`: disable windows; normal mode then requires `--bbox`.
- `--validate`: generate and track a deterministic synthetic clip.

Python uses `--list-trackers`; C++ uses `--list`. Invalid options, malformed or
out-of-frame boxes, missing inputs, and output failures return a clean nonzero
exit.

Validation generates an 80-frame, 640×360 clip containing a textured square on
a seeded noisy background. It initializes from frame zero and passes only when
the 79 subsequent updates have mean IoU ≥ 0.45 and at least 90% exceed 0.30
IoU. It writes the synthetic clip, an 80-frame annotated clip, and metrics
JSON, then prints `VALIDATION PASSED` or `VALIDATION FAILED`.

## Test

```shell
# Python: 23 tests, including all six tracker entry points.
python3 -m unittest discover -s python/tests -v

# C++: 17 tests with contrib and all downloaded models.
ctest --test-dir cpp/build --output-on-failure
```

The Python suite runs the real CLI headlessly from temporary working
directories. Missing DNN model files are explicit skips, while missing APIs,
corrupt models, and construction failures fail the suite. Each tracker test
checks the exact artifact set, nonempty files, readable 640×360 videos, 80
frames, metrics fields, thresholds, and zero lost frames.

CTest uses portable CMake wrappers to check the pass marker, exact artifact
manifest, nonempty files, readable 80-frame AVIs, metrics keys, frame/loss
contract, and clean failures for malformed options, values, bounding boxes,
missing input, and incomplete validation.
Download models before configuring CMake because DNN tracker tests are
registered only when their files exist at configure time.

For the acceptance matrix, run both commands with exact OpenCV 4.14.0 and
5.0.0 environments, use fresh CMake build directories, and also invoke every
tracker's `--validate --no-display` entry point from an unrelated working
directory.

## Compatibility notes

- All six trackers passed the same thresholds on exact OpenCV 4.14.0 and 5.0.0
  in Python and C++. Tracker scores can differ across OpenCV versions; passing
  semantics and artifact contracts—not four-decimal score equality—are the
  compatibility guarantee.
- OpenCV 5's new DNN graph engine prints `setPreferableTarget` warnings when the DNN trackers load; they are harmless and do not affect results.
- The C++ and Python synthetic clips share the same geometry, trajectory, and thresholds, but use different random generators, so their pixels (and exact IoU values) differ slightly across languages by design.
- Output videos preserve the input frame rate when the backend reports one and
  otherwise use a 30 FPS fallback.
- Performance is platform- and workload-specific; the FPS overlay measures
  tracker updates only.

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

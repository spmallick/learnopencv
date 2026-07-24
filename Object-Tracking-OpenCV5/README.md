# OpenCV 5 Object Trackers: A Complete Guide (C++/Python)

**This repository contains code for the [OpenCV 5 Object Trackers: A Complete Guide (C++/Python)](https://learnopencv.com/opencv5-object-trackers/) blogpost**.

One command-line application — implemented twice, in Python and in C++ — runs every single-object tracker that ships with OpenCV 5:

| Tracker | Type | Module | Model files |
| --- | --- | --- | --- |
| MIL | classical | main (`video`) | none |
| KCF | classical | contrib (`tracking`) | none |
| CSRT | classical | contrib (`tracking`) | none |
| DaSiamRPN | deep learning | main (`video`) | 3 ONNX files |
| NanoTrack v2 | deep learning | main (`video`) | 2 ONNX files |
| VitTrack | deep learning | main (`video`) | 1 ONNX file |

GOTURN is intentionally absent: OpenCV 5 removed its Caffe DNN importer, and with it the GOTURN tracker. The article covers the migration path.

## Supported versions

- OpenCV **4.9.0 – 5.x** (verified: Python on the **4.13.0.92** and **5.0.0.93** `opencv-contrib-python` wheels; C++ against an **OpenCV 5.0.0 + contrib** source build). The 4.9 floor exists because TrackerVit first shipped in OpenCV 4.9.0.
- Python 3.9+ with `opencv-contrib-python>=4.9,<6` and `numpy>=1.23,<3`.
- C++17, CMake 3.16+. KCF and CSRT require an OpenCV build that includes the opencv_contrib `tracking` module; without it the C++ example still builds and the two trackers report themselves unavailable.

## Directory Structure

```text
Object-Tracking-OpenCV5/
├── README.md
├── download_models.py
├── cpp/
│   ├── CMakeLists.txt
│   └── object_tracking.cpp
└── python/
    ├── requirements.txt
    ├── object_tracking.py
    └── tests/
        └── test_object_tracking.py
```

The `models/` directory is created next to `download_models.py` on first download; model files are not committed to the repository.

## Setup

Download the ONNX models for the three deep-learning trackers (the classical trackers need no files):

```shell
python3 download_models.py
```

The script verifies SHA-256 checksums where the upstream host publishes stable files and prints the computed hash for every download.

### Python

```shell
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### C++

```shell
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

To build against a specific OpenCV installation, point CMake at it:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/opt/opencv-5.0.0/lib/cmake/opencv5
```

## Run

Both implementations expose the same controls. Inputs may be a video path or a numeric camera index; bundled assets are resolved relative to the source location, so any current working directory works.

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
./cpp/build/object_tracking --tracker=mil --validate --no-display
```

Options common to both: `--bbox x,y,w,h` (initial box; drawn interactively when omitted), `--models-dir PATH`, `--output-dir PATH` (writes `tracked_<name>.avi` and `metrics_<name>.json`), `--max-frames N`, `--no-display` (headless; requires `--bbox` in normal mode), and `--validate`.

`--validate` generates a deterministic synthetic clip (textured square on a noisy background, seeded), initializes the tracker from the ground-truth box of frame zero, and passes only when mean IoU ≥ 0.45 and at least 90% of frames stay above 0.30 IoU. It prints `VALIDATION PASSED` or `VALIDATION FAILED` and exits accordingly.

## Tests

```shell
# Python: runs the real CLI in subprocesses, headless, from temp directories.
python3 -m unittest discover -s python/tests -v

# C++: one CTest regression per available tracker plus error-handling checks.
cd cpp/build && ctest --output-on-failure
```

Trackers that are unavailable in the current build (no contrib) or whose models are not downloaded are skipped with an explicit reason. CTest registers the DNN-tracker tests only when the model files exist at configure time.

## Compatibility notes

- The same sources run unchanged on OpenCV 4.9+ and 5.x. On the reference synthetic clip, per-tracker mean IoU agreed to four decimal places between OpenCV 4.x and 5.0 in the Python matrix.
- OpenCV 5's new DNN graph engine prints `setPreferableTarget` warnings when the DNN trackers load; they are harmless and do not affect results.
- The C++ and Python synthetic clips share the same geometry, trajectory, and thresholds, but use different random generators, so their pixels (and exact IoU values) differ slightly across languages by design.
- Performance is platform- and workload-specific; the FPS overlay measures tracker updates only.

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

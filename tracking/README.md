# [Object Tracking using OpenCV (C++/Python)](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)

[<img src="https://learnopencv.com/wp-content/uploads/2017/02/real-time-face-tracking.gif" alt="A bounding box follows Charlie Chaplin through a video.">](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/object-tracking-opencv-2026.07.29/tracking-2026.07.29.zip)

[Verify the standalone ZIP with its SHA-256 checksum.](https://github.com/spmallick/learnopencv/releases/download/object-tracking-opencv-2026.07.29/tracking-2026.07.29.zip.sha256)

This refreshed companion project demonstrates OpenCV's current single-object tracking API in Python 3 and C++17. It keeps the classic MIL, KCF, and CSRT paths and adds the model-backed DaSiamRPN, NanoTrack v2, and VitTrack APIs. It is headless by default, validates every input, and emits a machine-readable summary for tests and automation.

Run every command below from this `tracking/` directory. In the LearnOpenCV
repository, start with `cd tracking`; the standalone release extracts to the
same top-level `tracking/` directory.

## Requirements

- Python 3.10 or newer; the publication matrix used Python 3.14.
- OpenCV 4.14.0 or 5.0.0 with the `video` and `dnn` modules. OpenCV 4.13.0
  remains covered by an additional regression run.
- CMake 3.16 or newer and a C++17 compiler for the native example.
- `ffmpeg` and `ffprobe` only when rendering comparison MP4s.

## Compatibility

| Capability | Tested exact OpenCV 4.14 builds | Tested exact OpenCV 5.0 builds |
|---|---|---|
| MIL tracker | Available; default | Available; default |
| CSRT and KCF | Present in the tested 4.14 contrib Python build; capability-gated in C++ | Absent from the exact 5.0 builds tested here; other contrib-enabled builds and bindings may expose them |
| DaSiamRPN | Available when the `video` and `dnn` modules plus three ONNX files are present | Same |
| NanoTrack v2 | Available when the `video` and `dnn` modules plus two ONNX files are present | Same |
| VitTrack | Available in OpenCV 4.9+ when the `video` and `dnn` modules plus one ONNX file are present | Same |
| Interactive ROI | `--select-roi` | `--select-roi` |
| Headless fixed box | `--bbox=x,y,width,height` | `--bbox=x,y,width,height` |

The Python factory checks the tracker factories and classes actually exposed by the installed package. The C++ source conditionally includes the contrib tracking header whenever it is available. MIL and the three model-backed trackers live in the main `video` module; the model-backed implementations also require `dnn`. This capability check is more reliable than assuming that every package with the same major version contains the same trackers.

## Download the DNN tracker models

Classic MIL, KCF, and CSRT do not need external models. Download one model set or all three from immutable upstream revisions:

```bash
python download_models.py --tracker nano
python download_models.py --tracker vit
python download_models.py --tracker dasiamrpn

# Or fetch all six files:
python download_models.py --tracker all
```

The downloader writes to `models/` by default, streams into a temporary file, and checks both the exact size and SHA-256 digest before replacing anything. It can also target another directory with `--models-dir /path/to/models`; pass the same directory to `tracker.py` or `object_tracker`.

| Tracker | Required local files | Total download |
|---|---|---:|
| DaSiamRPN | `dasiamrpn_model.onnx`, `dasiamrpn_kernel_cls1.onnx`, `dasiamrpn_kernel_r1.onnx` | 154.35 MiB |
| NanoTrack v2 | `nanotrack_backbone_sim.onnx`, `nanotrack_head_sim.onnx` | 1.70 MiB |
| VitTrack | `object_tracking_vittrack_2023sep.onnx` | 0.68 MiB |

The DaSiamRPN and VitTrack files come from pinned revisions of the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo). NanoTrack v2 comes from the model directory referenced by OpenCV's [TrackerNano API](https://docs.opencv.org/4.x/d8/d69/classcv_1_1TrackerNano.html). Review the upstream model licenses before redistributing the binaries.

## Run the Python example

```bash
python -m pip install -r requirements.txt
python tracker.py --tracker MIL --max-frames 60 \
  --output tracked.avi --snapshot tracked.png
```

The included Chaplin clip uses the tested initial box `287,23,86,320`. For another video, either pass `--bbox` or add `--select-roi --display`.

After downloading the models, run each deep-learning tracker with the same CLI:

```bash
python tracker.py --tracker DASIAMRPN --models-dir models \
  --bbox 287,23,86,320 --max-frames 60 --output dasiamrpn.avi

python tracker.py --tracker NANO --models-dir models \
  --bbox 287,23,86,320 --max-frames 60 --output nano.avi

# VitTrack is more stable on this clip when the initial box covers the full body.
python tracker.py --tracker VIT --models-dir models \
  --bbox 250,15,180,340 --max-frames 60 --output vit.avi
```

These are still single-object trackers, not detectors. You must provide the initial target box. A successful update or high internal score does not prove that the box still covers the correct identity.

## Render a tracker comparison video

`render_tracker_comparison.py` initializes every requested tracker on the same
source frame and ROI. By default, it creates a 1280x720 H.264 MP4 with a
side-by-side section followed by one full-screen replay per tracker. Pass
`--together-only` to keep only the simultaneous comparison grid, with no title
card or sequential replays. Every run also creates a JSON file containing the
exact source-frame range, input fingerprint, trajectories, and output
verification.

The bundled Charlie Chaplin comparison uses `videos/chaplin.mp4`. Its SHA-256
is `ce6154a3a0223861cdc6b1a0301e650e55cff7132e586845924699b9ea6162dc`;
the source is 640x360 at OpenCV's reported 29.719556780471343 fps. Although
the container header reports 172 frames, exactly 150 frames decode, numbered
0 through 149. The comparison shows the turn away from the camera, rear view,
return, and left-edge boundary clipping without claiming that a returned box
still covers the target correctly.

The classic single-tracker example keeps its tested narrow ROI
`287,23,86,320` unchanged. This comparison alone uses the wider full-body ROI
`250,15,180,340` as a fair shared initialization for all four trackers,
consistent with the documented VitTrack guidance above.

```bash
python render_tracker_comparison.py \
  --input videos/chaplin.mp4 \
  --output charlie-chaplin-opencv5-four-trackers.mp4 \
  --models-dir models \
  --trackers MIL,DASIAMRPN,NANO,VIT \
  --start-frame 0 --end-frame 149 \
  --bbox 250,15,180,340 \
  --scene-title "Charlie Chaplin Comparison" \
  --together-only
```

The verified street result uses [street footage by tiburi on
Pixabay](https://pixabay.com/videos/street-walk-urban-city-walking-5025/),
saved locally as `pixabay-street-5025-182666664-large.mp4`. The exact source is
1920x1080, 30000/1001 fps, and 330 frames; its SHA-256 is
`f4a7be9346410b79dc0b526ab4ec5c0c58589865b48136d2b3f7adbd240f50d7`.
This is intentionally a limitation example: people and bicycles occlude the
selected pedestrian, and all four single-object trackers have limited success
once the target becomes ambiguous.

```bash
python render_tracker_comparison.py \
  --input pixabay-street-5025-182666664-large.mp4 \
  --output street-scene-opencv5-four-trackers.mp4 \
  --models-dir models \
  --trackers MIL,DASIAMRPN,NANO,VIT \
  --start-frame 0 --end-frame 329 \
  --bbox 351,321,57,249 \
  --scene-title "Street Occlusion Limitation" \
  --together-only
```

The verified car result uses [racing footage by DistillVideos on
Pixabay](https://pixabay.com/videos/car-racing-motor-sports-action-74/),
saved locally as `pixabay-car-74.mp4`. The exact source is 1280x720,
24000/1001 fps, and 907 frames; its SHA-256 is
`cc789c74e0c007d531ea872f76b243187edf984a3eeaa4074b9534679b03f82c`.
This is also a limitation example: fast motion and rapid appearance,
viewpoint, and scale changes challenge the trackers during the original
65-frame sequence.

```bash
python render_tracker_comparison.py \
  --input pixabay-car-74.mp4 \
  --output race-car-opencv5-four-trackers.mp4 \
  --models-dir models \
  --trackers MIL,DASIAMRPN,NANO,VIT \
  --start-frame 256 --end-frame 320 \
  --bbox 734,420,300,100 \
  --scene-title "Race Car Motion Limitation" \
  --together-only
```

Each command places MIL, DaSiamRPN, NanoTrack, and VitTrack in the same 2x2
panel for exactly the selected source frames. Both frame arguments are
zero-based and inclusive; omit `--end-frame` to use the rest of a video. Run
`python render_tracker_comparison.py --help` for encoding and overwrite
options. The renderer requires `ffmpeg` and `ffprobe`. Its status labels
distinguish the initial ROI, an update that returned a box, a returned box that
extends off frame, and an update that returned no box. A returned box is not a
claim that the tracker still covers the correct object.

## Build and run C++17

```bash
cmake -S . -B build -DBUILD_TESTING=ON \
  -DTRACKING_MODELS_DIR="$PWD/models"
cmake --build build --parallel
./build/object_tracker --tracker=MIL --max-frames=60 \
  --output=tracked.avi --snapshot=tracked.png
ctest --test-dir build --output-on-failure
```

To select a particular OpenCV installation, pass its package directory:

```bash
cmake -S . -B build-opencv-5 \
  -DOpenCV_DIR=/path/to/opencv-5/lib/cmake/opencv5 \
  -DBUILD_TESTING=ON \
  -DTRACKING_MODELS_DIR="$PWD/models"
```

Use `--tracker=DASIAMRPN`, `--tracker=NANO`, or `--tracker=VIT` and add `--models-dir=models` to run the C++ model-backed paths. CMake requires OpenCV's `dnn` component so a build that cannot load these networks fails during configuration instead of later in the tutorial.

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q

# Include the downloaded-model integration tests:
TRACKING_MODELS_DIR="$PWD/models" python -m pytest tests -q
```

The offline tests validate parsing, missing-input and missing-model errors, all six pinned model records, atomic download behavior, classic tracker creation, five successful updates on the bundled video, the annotated output video, and the saved result frame. When `TRACKING_MODELS_DIR` is set, the suite also creates DaSiamRPN, NanoTrack, and VitTrack and processes five real updates with each one.

With the verified model directory enabled, all 17 Python tests passed on exact OpenCV 4.14.0 and 5.0.0, with an additional 4.13.0 regression run. The suite includes coverage for the four-panel-only timeline and failure-injection coverage that verifies the comparison renderer restores the prior MP4/JSON pair when final installation is interrupted. The native C++ build and all four CTests (MIL plus the three model-backed trackers) also passed against exact OpenCV 4.14.0 and 5.0.0 builds containing `dnn`, with the same four CTests repeated on 4.13.0.

## Standalone release contents

The immutable release ZIP contains exactly these 15 tracked files under one
top-level `tracking/` directory. ONNX models, caches, generated videos, and
build products are not included.

```text
tracking/
├── .gitignore
├── CMakeLists.txt
├── README.md
├── download_models.py
├── models/
│   ├── .gitignore
│   └── README.md
├── render_tracker_comparison.py
├── requirements-test.txt
├── requirements.txt
├── tests/
│   ├── test_download_models.py
│   ├── test_render_tracker_comparison.py
│   └── test_tracker.py
├── tracker.cpp
├── tracker.py
└── videos/
    └── chaplin.mp4
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

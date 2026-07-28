# OpenCV 5 in C++: How To Detect, Pose & Segment Objects

**This repository contains the code for the LearnOpenCV blog post [OpenCV 5 in C++: How To Detect, Pose & Segment Objects](https://learnopencv.com/opencv-5-in-c-how-to-detect-pose-segment-objects/).**

[![OpenCV 5 in C++: How To Detect, Pose & Segment Objects](opencv5-cpp-football-vision-thumbnail.jpg)](https://learnopencv.com/opencv-5-in-c-how-to-detect-pose-segment-objects/)

Companion code for the [LearnOpenCV post](https://learnopencv.com/opencv-5-in-c-how-to-detect-pose-segment-objects/) on running **YOLO26** through the **OpenCV 5 DNN module** in **C++**, on the **CPU**. This is Part 2 of the series (Part 1 was Python). Here everything is C++, and the same NMS-free pipeline is extended from plain object detection to **team splitting, pose estimation and instance segmentation**.

Everything runs on the **CPU only**. There is no CUDA and no GPU requirement, so
you can build and run all of it on an ordinary laptop. No PyTorch or Python is
needed to *run* the demos; Python is used only once, to export the models to
ONNX.

## What is in this repository

| Path | What it is | Why it is here |
|---|---|---|
| `src/` | All the C++ sources | The actual implementation you build |
| `scripts/` | Python model-export scripts | Turn YOLO26 weights into the ONNX files (run once) |
| `assets/` | Source images and videos | The exact inputs the demos run on |
| `CMakeLists.txt` | Build config for the demos | Builds the six executables |
| `build_opencv5.sh` | Builds OpenCV 5 from source | The C++ dev files are not on pip yet |
| `reproduce.sh` | Regenerates every blog result | One command, from `assets/` to annotated outputs |

### `src/` (the C++ code)

```
yolo26_dnn.hpp / .cpp   shared core: engine select, letterbox, NMS-free
                        detect / pose / seg parsing, and all the drawing
cli.hpp                 tiny "--key value" argument parser
detect_image.cpp        object detection on a single image
detect_video.cpp        object detection on a video (+ FPS, end-to-end timing)
detect_teams.cpp        split players into two teams by shirt colour (k-means)
detect_pose.cpp         YOLO26-pose, 17-point skeletons
detect_seg.cpp          YOLO26-seg, per-instance masks
benchmark_engines.cpp   time the new vs the classic DNN engine
```

### `scripts/` (make the models)

```
export_yolo26_onnx.py   yolo26n.pt      -> yolo26n_640.onnx, yolo26n_1280.onnx
export_variants.py      yolo26n-seg.pt  -> yolo26n_seg_640.onnx
                        yolo26n-pose.pt -> yolo26n_pose_640.onnx
```

The weights download automatically through Ultralytics the first time you run
these. Static input shape, opset 12, no embedded NMS, which is what the OpenCV 5
DNN importer expects.

## What is deliberately NOT in this repository

- **`models/`** (the `.pt` weights and `.onnx` files). They are generated, not
  shipped. `reproduce.sh` exports them automatically on first run, or you can run
  the export scripts yourself. This keeps the repo small and always current.
- **`outputs/`** (the annotated images and videos). You produce your own by
  running the code on `assets/`, so there is nothing to ship.
- **Build artifacts** (`build/`, the compiled `.exe` binaries, and the OpenCV
  source/build trees). These are rebuilt locally.

All of the above are in `.gitignore`.

## Prerequisites

- A **C++17** compiler, **CMake**, and **Ninja**
- The **FFmpeg development libraries** (for reading/writing MP4)
- **Python 3** with `ultralytics`, `onnx`, `onnxslim` (only to export the models)

On MSYS2 (Windows): `pacman -S mingw-w64-ucrt-x86_64-{cmake,ninja,ffmpeg}`
On Debian/Ubuntu: `apt install cmake ninja-build libavcodec-dev libavformat-dev libswscale-dev`

## How to build and run

```bash
# 1. Build OpenCV 5 from source (CPU, DNN). Prints an OpenCV_DIR path at the end.
./build_opencv5.sh

# 2. Build the six demos against that OpenCV.
cmake -G Ninja -B build -DOpenCV_DIR=$(pwd)/opencv5_install/lib/cmake/opencv5
cmake --build build

# 3. Reproduce every result. First run auto-exports the models, then runs
#    each demo on assets/ and writes annotated results into outputs/.
pip install ultralytics onnx onnxslim
./reproduce.sh
```

On Windows/MinGW use `-DOpenCV_DIR=$(pwd)/opencv5_install/x64/mingw/lib` in step 2.

## Running a single demo

Run each of these **from the repo root**. Every demo also runs with **no
arguments at all** (it falls back to the exact defaults shown here), so once the
models exist you can just run e.g. `build/detect_image`.

```bash
# detection
build/detect_image --model models/yolo26n_640.onnx --imgsz 640 \
    --source assets/images/team_lineup.jpg --engine new --out outputs/image_detected.jpg

# pose (image or video, auto-detected)
build/detect_pose  --model models/yolo26n_pose_640.onnx \
    --source assets/images/aerial_duel.jpg --out outputs/pose.jpg

# instance segmentation
build/detect_seg   --model models/yolo26n_seg_640.onnx \
    --source assets/images/goal_celebration.jpg --out outputs/seg.jpg

# team split (video)
build/detect_teams --model models/yolo26n_640.onnx --imgsz 640 --teams 2 \
    --source assets/videos/stadium_wide.mp4 --out outputs/team_split.mp4

# engine benchmark
build/benchmark_engines --model models/yolo26n_640.onnx --imgsz 640 \
    --source assets/images/team_lineup.jpg --runs 30
```

`--engine` takes `auto` (default), `new`, or `classic`. `--imgsz` must match the
exported model, since the ONNX input shape is static.

## Notes

- YOLO26 is NMS-free. The ONNX outputs are already finalized: detection
  `[1,300,6]`, pose `[1,300,57]` (6 + 17 keypoints x 3), segmentation
  `[1,300,38]` (6 + 32 mask coeffs) plus a `[1,32,160,160]` prototype tensor.
- A `cv::dnn::Net` is not thread-safe; give each thread its own network.
- On Windows/MinGW the vendored MLAS asm kernels do not build (they target the
  System V ABI); OpenCV falls back to its built-in SGEMM. Detection is correct,
  just a little slower than an MSVC build.
- Labels read "player" and "ball" (a football-friendly display-name mapping); the
  underlying COCO classes are unchanged.

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

# [Object Tracking using OpenCV (C++/Python)](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)

[<img src="https://learnopencv.com/wp-content/uploads/2017/02/real-time-face-tracking.gif" alt="A bounding box follows Charlie Chaplin through a video.">](https://learnopencv.com/object-tracking-using-opencv-cpp-python/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/object-tracking-opencv-2026.07.27/tracking-2026.07.27.zip)

This refreshed companion project demonstrates OpenCV's current single-object tracking API in Python 3 and C++17. It is headless by default, validates every input, and emits a machine-readable summary for tests and automation.

## Compatibility

| Capability | OpenCV 4 | OpenCV 5 |
|---|---|---|
| MIL tracker | Supported; default | Supported; default |
| CSRT and KCF | Available with an OpenCV contrib build | Removed from the 5.0 API surface |
| Interactive ROI | `--select-roi` | `--select-roi` |
| Headless fixed box | `--bbox=x,y,width,height` | `--bbox=x,y,width,height` |

The Python factory checks both the OpenCV 5 class-based `TrackerMIL.create()` API and OpenCV 4 factory functions. The C++ source conditionally includes the OpenCV 4 contrib tracking header while keeping MIL available through the main video module.

## Run the Python example

```bash
python -m pip install -r requirements.txt
python tracker.py --tracker MIL --max-frames 60 \
  --output tracked.avi --snapshot tracked.png
```

The included Chaplin clip uses the tested initial box `287,23,86,320`. For another video, either pass `--bbox` or add `--select-roi --display`.

## Build and run C++17

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
./build/object_tracker --tracker=MIL --max-frames=60 \
  --output=tracked.avi --snapshot=tracked.png
ctest --test-dir build --output-on-failure
```

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

The smoke tests validate parsing, missing-input errors, tracker creation, five successful updates on the bundled video, the annotated output video, and the saved result frame. The project was tested with Python OpenCV 4.13 and exact OpenCV 5.0.0, plus exact native C++ OpenCV 4.13.0 and exact 5.0.0.


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

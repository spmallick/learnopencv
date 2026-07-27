# [Stereo Camera Depth Estimation With OpenCV (Python/C++)](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2021/06/04093439/Pg5-Depth-perception-using-stereo-camera-FeatureImage-768x432.jpg" alt="Stereo camera depth estimation and obstacle avoidance with OpenCV" width="800">](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download the stereo depth companion code" width="200">](https://github.com/spmallick/learnopencv/releases/download/stereo-depth-opencv-2026.07.26/Depth-Perception-Using-Stereo-Camera-2026.07.26.zip)

This companion uses OpenCV `StereoBM` to rectify a stereo pair, estimate
pixel disparity, calibrate a disparity-to-depth model, and highlight nearby
obstacles. Camera-independent Python and C++ functions are covered by
deterministic tests; real-camera scripts remain available as optional smoke
tests.

## Requirements

- Python 3.10–3.14 with OpenCV 4.13 or newer
- CMake 3.16 or newer
- A C++17 compiler
- OpenCV 4 or OpenCV 5 development libraries
- A synchronized stereo camera only for the interactive examples

```bash
python -m pip install -r python/requirements.txt
```

All default map and configuration paths are resolved relative to this
directory, not the current working directory.

## Headless image-pair workflow

Provide a 640×480 pair from the calibrated rig:

```bash
python python/stereo_depth.py \
  --left path/to/left.png \
  --right path/to/right.png
```

The command writes rectified inputs, disparity, and depth visualizations under
`outputs/`. Add `--already-rectified` when the pair has already been
rectified.

The equivalent C++ executable is built as `stereo_depth_cli`.

## Deterministic tests

The tests use generated textured stereo pairs, known disparity/depth models,
the tracked 640×480 rectification maps, invalid-disparity cases, configuration
round trips, and a synthetic obstacle:

```bash
python -m pytest -q python/tests/test_stereo_depth.py

cmake -S cpp -B cpp/build -DBUILD_TESTING=ON
cmake --build cpp/build --config Release
ctest --test-dir cpp/build --output-on-failure -C Release
```

## Optional real-rig workflows

Tune `StereoBM` without mutating the input XML:

```bash
python python/disparity_params_gui.py \
  --save-config data/depth_estimation_params_py_updated.xml
```

Fit both scale and offset in
`depth = scale / (disparity - minDisparity) + offset`:

```bash
python python/disparity2depth_calib.py
```

Run obstacle visualization:

```bash
python python/obstacle_avoidance.py
```

For a bounded, non-GUI hardware smoke test:

```bash
python python/disparity_params_gui.py \
  --no-display --max-frames 60 --output outputs/real-rig-disparity.png
```

The C++ executables accept the corresponding `--left-camera`,
`--right-camera`, `--maps`, `--config`, `--max-frames`, and output options.
They use `grab` on both cameras before `retrieve` to reduce capture skew.

Legacy XML files store `M` against normalized disparity. The loaders migrate
that value to a pixel-disparity scale in memory. Newly written configuration
files store explicit `depthScale` and `depthOffset` fields.


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

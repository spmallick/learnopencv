# [Understanding Lens Distortion with OpenCV](https://learnopencv.com/understanding-lens-distortion/)

[![Understanding lens distortion with OpenCV](https://cdn.learnopencv.com/wp-content/uploads/2020/04/04095232/Understanding-lens-distortion-e1586524454617.png)](https://learnopencv.com/understanding-lens-distortion/)

This folder contains the tested Python and C++ companion code for the
[LearnOpenCV article](https://learnopencv.com/understanding-lens-distortion/).
It calibrates a camera from the included checkerboard photographs, reports the
reprojection error and valid-pixel region, and saves corrected images using both
OpenCV undistortion paths.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/lens-distortion-opencv-2026.07.26/UnderstandingLensDistortion-2026.07.26.zip)

## What was corrected

- Image dimensions are passed to OpenCV as `(width, height)`, not
  `(height, width)`.
- `cv::undistort` and `cv2.undistort` receive the original camera matrix and
  use the optimized matrix only as the output projection.
- The first image with a successful checkerboard detection is selected
  explicitly instead of relying on the last loop iteration.
- Both implementations compute a reprojection RMSE, expose the valid-pixel ROI,
  validate every input image, and run headlessly by default.
- The direct and precomputed-map correction paths are tested for numerical
  agreement.

## Requirements

- Python 3.10 or newer (tested with Python 3.14)
- OpenCV 4.13 for Python
- CMake 3.16 or newer and a C++17 compiler
- OpenCV 4 or OpenCV 5 development libraries for C++

Install the Python dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

## Run the Python example

The default image and output paths are relative to `Undistort.py`, so this
works from any current directory:

```bash
python Undistort.py --require-all
```

Results are saved under `outputs/`:

- `calibration-corners.jpg`
- `undistorted-direct.jpg`
- `undistorted-remap.jpg`
- `calibration.yml`

Use `--crop` to crop corrected images to the valid-pixel ROI. Add `--show` only
when a desktop display is available.

## Build and run the C++ example

Keep build products outside this source directory:

```bash
cmake -S . -B /tmp/lens-distortion-build \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/lens-distortion-build
/tmp/lens-distortion-build/Undistort --require-all
```

The C++ executable uses the same script-independent defaults and output names
as the Python version.

## Run the tests

The tests require every one of the 41 included checkerboard images to be
detected. They check calibration stability, the reprojection error, ROI,
direct-versus-remap parity, cropped output, and invalid inputs.

```bash
python -m pytest -q tests/test_lens_distortion.py
ctest --test-dir /tmp/lens-distortion-build --output-on-failure
```

The expected OpenCV calibration RMS is approximately `0.2603` pixels for this
dataset. Tests use version-tolerant bounds for calibration coefficients, ROI,
and interpolation differences so they remain useful across OpenCV 4 and 5.

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

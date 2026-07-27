# [Otsu’s Thresholding with OpenCV](https://learnopencv.com/otsu-thresholding-with-opencv/)

[<img src="https://cdn.learnopencv.com/wp-content/uploads/2020/06/04094657/otsu_res_cover.png" alt="Otsu thresholding separates the foreground and background of an image" width="800">](https://learnopencv.com/otsu-thresholding-with-opencv/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download the Otsu thresholding companion code" width="200">](https://github.com/spmallick/learnopencv/releases/download/otsu-thresholding-opencv-2026.07.26/otsu-method-2026.07.26.zip)

This companion implements Otsu thresholding from scratch and compares its
threshold and binary output with OpenCV. Both Python and C++ paths are
headless, validate their inputs, and resolve default assets relative to this
directory.

## Requirements

- Python 3.10–3.14 with OpenCV 4.13 or newer
- CMake 3.16 or newer
- A C++17 compiler
- OpenCV 4 or OpenCV 5 development libraries for the C++ examples

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Python

Run the custom implementation and the OpenCV reference:

```bash
python otsu_implementation.py
python otsu_method.py
```

Both commands write results under `outputs/`. Use `--help` to select another
input or output location. No display server is required.

Run the deterministic regression tests:

```bash
python -m pytest -q tests/test_otsu.py
```

## C++

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
./build/otsu_implementation
./build/otsu_method
```

The regression test checks the real `boat.jpg` fixture, synthetic bimodal
data, constant images, invalid inputs, and pixel-for-pixel parity with
OpenCV. For `boat.jpg`, both implementations select threshold `132`.

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

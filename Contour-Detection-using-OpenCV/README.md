# Contour Detection using OpenCV

**This repository contains code for [Contour Detection using OpenCV](https://learnopencv.com/contour-detection-using-opencv-python-c/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://github.com/spmallick/learnopencv/releases/download/contour-detection-opencv-2026.07.22/Contour-Detection-using-OpenCV-2026.07.22.zip)

## Directory Structure

**All the code files and folders follow the following structure.**

```
├── cpp
│   ├── channel_experiments
│   │   ├── channel_experiments.cpp
│   │   └── CMakeLists.txt
│   ├── contour_approximations
│   │   ├── CMakeLists.txt
│   │   └── contour_approx.cpp
│   └── contour_extraction
│       ├── CMakeLists.txt
│       └── contour_extraction.cpp
├── input
│   ├── custom_colors.jpg
│   ├── image_1.jpg
│   └── image_2.jpg
├── python
│   ├── channel_experiments
│   │   └── channel_experiments.py
│   ├── contour_approximations
│   │   └── contour_approx.py
│   ├── contour_extraction
│   │   └── contour_extraction.py
│   ├── tests
│   │   └── test_examples.py
│   └── requirements.txt
└── README.md
```



## Requirements

The examples support OpenCV 4.x and OpenCV 5.x. Python requires Python 3.9 or
newer. The C++ examples use C++17 and CMake 3.16 or newer.

## Instructions

### Python

Install a supported OpenCV version and run the commands below from the project
directory. The examples resolve their bundled input images from the script
location, so absolute script paths also work from another current directory.

```shell
python3 -m pip install -r python/requirements.txt
python3 python/channel_experiments/channel_experiments.py
python3 python/contour_approximations/contour_approx.py
python3 python/contour_extraction/contour_extraction.py
```

Every Python example accepts `--no-display` for headless environments,
`--output-dir PATH`, and `--validate` for regression checks. Run the complete
Python regression suite with:

```shell
python3 -m unittest discover -s python/tests -v
```

Use `--input PATH` to override the image for the channel and retrieval-mode
examples. The approximation example accepts `--image1 PATH --image2 PATH`.

### C++

Configure, build, and test each C++ example independently. For example:

```shell
cmake -S cpp/channel_experiments -B cpp/channel_experiments/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/channel_experiments/build --config Release
ctest --test-dir cpp/channel_experiments/build --output-on-failure
./cpp/channel_experiments/build/channel_experiments
```

Replace `channel_experiments` with `contour_approximations` or
`contour_extraction` for the other examples. The executables support the same
`--no-display`, `--output-dir`, and `--validate` options as their Python
counterparts.

To select a particular OpenCV installation, pass its package directory while
configuring:

```shell
cmake -S cpp/contour_extraction -B cpp/contour_extraction/build \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5
```

### `findContours` performance in OpenCV 4.14 and 5

OpenCV 4.14 introduced the parallel TRUCO implementation that is also used by
OpenCV 5. It is selected for an 8-bit single-channel input with `RETR_LIST`
when hierarchy output is not requested. The four-argument C++ overload makes
that explicit:

```cpp
std::vector<std::vector<cv::Point>> contours;
cv::findContours(binary, contours, cv::RETR_LIST,
                 cv::CHAIN_APPROX_SIMPLE);
```

Passing `cv::noArray()` as the hierarchy output to the hierarchy-accepting
overload also leaves hierarchy unrequested and can select the same fast path.

The tutorial examples intentionally request hierarchy or use another retrieval
mode while explaining contour relationships, so changing those calls would
change their documented behavior. The current Python binding also returns
hierarchy and does not expose the no-hierarchy fast path.

In Release builds on an Apple M5 Pro, the no-hierarchy overload was about
1.3–1.6 times as fast as the hierarchy-producing overload on the two tutorial
images. OpenCV 4.14 and 5 had essentially the same fast-path performance because
both contain TRUCO. On a 6-core/12-thread Intel Core i7-6850K Linux machine, the
same comparison was about 2.7–3.0 times with the default 12-thread pthreads
backend and 1.38–1.50 times in a single-thread sensitivity run. This spread is
why the result must be treated as platform- and workload-specific; benchmark the
exact call on your deployment hardware. See the
[current `findContours` API documentation](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
for the activation conditions, and the
[TRUCO implementation pull request](https://github.com/opencv/opencv/pull/28773)
for its development history and original benchmark discussion. The benchmark
fixtures produced the same contour sets, but outer-vector ordering is not an API
guarantee; sort or match contours explicitly when order matters.

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

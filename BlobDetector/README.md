# [Blob Detection Using OpenCV in Python and C++](https://learnopencv.com/blob-detection-using-opencv-python-c/)

[<img src="https://learnopencv.com/wp-content/uploads/2015/02/BlobTest.jpg" alt="OpenCV SimpleBlobDetector results" width="768">](https://learnopencv.com/blob-detection-using-opencv-python-c/)

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download code" width="200">](https://github.com/spmallick/learnopencv/releases/download/blob-detector-v2026.07.26/BlobDetector-2026.07.26.zip)

This directory contains the tested companion code for the LearnOpenCV article. Both implementations read an explicit image, detect blobs with the same `SimpleBlobDetector` parameters, write a visualization, and run without a desktop display by default.

## Tested versions

- Python 3.14, OpenCV Python 4.13, and NumPy 2.4
- C++17, CMake 3.29, and OpenCV C++ 4.12

OpenCV 4.10 through 5.x is supported.

The C++ build selects OpenCV 4's `features2d` module or OpenCV 5's renamed `features` module after detecting the installed major version. The local OpenCV 4 path and the exact OpenCV 5.0 Python and C++ test matrix are verified.

## Python

Install the dependencies and run the example from this directory:

```bash
python -m pip install -r requirements.txt
python blob.py --input blob.jpg --output blob-keypoints.png
```

Add `--display` to open the result in an OpenCV window.

## C++

```bash
cmake -S . -B build
cmake --build build
./build/blob_detector --input blob.jpg --output blob-keypoints-cpp.png
```

## Expected output

With the included `blob.jpg`, both implementations report:

```text
Detected 16 blobs.
```

The output is a color image with one red circle per blob. The circle diameter represents the scale reported by `SimpleBlobDetector`.

## Automated tests

```bash
python -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
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

# [Super Resolution in OpenCV](https://learnopencv.com/super-resolution-in-opencv/)

This directory contains the tested Python and C++ companion code for the
[LearnOpenCV article](https://learnopencv.com/super-resolution-in-opencv/).

<a href="https://learnopencv.com/super-resolution-in-opencv/"><img src="https://cdn.learnopencv.com/wp-content/uploads/2021/06/04093440/PG-5-Super-Resolution-in-OpenCV-768x432.jpg" alt="Super Resolution in OpenCV" width="900"></a>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://github.com/spmallick/learnopencv/releases/download/super-resolution-v2026.07.26/Super-Resolution-in-OpenCV-2026.07.26.zip)

The example uses OpenCV's `dnn_superres` module and the ESPCN x4 TensorFlow
model. The model downloader is pinned to an immutable upstream revision and
verifies the file's SHA-256 checksum before installing it.

The model is redistributed unchanged from `fannymonori/TF-ESPCN` at commit
`a899033b12cd0400454fb5777600883a9d7e86c3` under the Apache License 2.0.
`models/README.md` records its upstream path, size, and checksum, and
`models/TF-ESPCN-LICENSE` contains the exact upstream license text.

## Requirements

- Python 3.10 or newer
- OpenCV 4.8 or newer with `dnn_superres`
- CMake 3.16 or newer and a C++17 compiler for the C++ example

Install the Python dependency and the verified model:

```bash
python3 -m pip install -r requirements.txt
python3 download_model.py
```

The bundled `image.png` is used by default. All paths can be overridden, so
the scripts also work when launched from another directory.

## Python

```bash
python3 super_res.py
```

An explicit invocation looks like this:

```bash
python3 super_res.py \
  --input image.png \
  --model models/ESPCN_x4.pb \
  --algorithm espcn \
  --scale 4 \
  --output output.png
```

## C++

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/SuperResolution \
  --input=image.png \
  --model=models/ESPCN_x4.pb \
  --algorithm=espcn \
  --scale=4 \
  --output=output-cpp.png
```

## Tests

The inference test uses the same downloaded and verified model:

```bash
python3 download_model.py
OPENCV_SUPERRES_MODEL=models/ESPCN_x4.pb \
  python3 -m unittest discover -s tests -v

cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

The tests verify configuration errors, missing-model behavior, actual ESPCN
inference, the x4 output dimensions, and output image readability. The CTest
case runs the compiled program through real ESPCN inference and requires the
reported `600x400 -> 2400x1600` dimensions.

The compatibility matrix passed Python 5/5 and CTest 1/1 against the official
OpenCV 5.0.0 tag. Local OpenCV 4 validation passed Python 5/5 with 4.13.0 and
CTest 1/1 with native 4.12.0.

For other supported architectures, download a compatible `.pb` file and pass
matching `--algorithm`, `--scale`, and `--model` values. OpenCV documents
EDSR, ESPCN, FSRCNN, and LapSRN in its
[DNN super-resolution module](https://docs.opencv.org/5.0/extra_modules/dnn_superres.html).

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

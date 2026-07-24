# Center of a blob using OpenCV (Python/C++)

This directory contains the Python and C++ examples for [Find the Center of a Blob using OpenCV](https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/).

Download the immutable, versioned
[CenterofBlob.zip](https://github.com/spmallick/learnopencv/releases/download/center-of-blob-opencv-2026.07.23/CenterofBlob.zip)
bundle and its
[SHA-256 checksum](https://github.com/spmallick/learnopencv/releases/download/center-of-blob-opencv-2026.07.23/CenterofBlob.zip.sha256).
The ZIP contains exactly one top-level `CenterofBlob/` directory with the files
shown in the directory layout below.

The migration is tested with OpenCV 4.14.0 and OpenCV 5.0.0. The Python examples require Python 3.9 or newer; the C++ examples require CMake 3.16 or newer and C++17. OpenCV 5 moved `moments` and `contourArea` into its `geometry` module, so the source and CMake build select the correct headers and components for each supported major version.

![Center of a blob](https://learnopencv.com/wp-content/uploads/2018/07/single-blob-image-768x307.png)

## Important correctness correction

The original single-blob example used `THRESH_BINARY` on a dark shape and therefore calculated the moment of the bright background, not the dark blob. The bundled circle happens to share its background's center, so the old program still printed `(112, 112)` and concealed the mistake. The corrected default uses `THRESH_BINARY_INV` and validates both the centroid and foreground-pixel count.

The original multiple-blob C++ example ran Canny edge detection followed by `RETR_TREE`. That measured nested boundaries on both sides of an edge instead of returning one filled foreground region per visible blob. The upgraded Python and C++ examples threshold the intended foreground and use `RETR_EXTERNAL`, producing exactly the six visible shapes. These are intentional correctness fixes, not OpenCV 5 behavior changes.

## How the centroid is calculated

Each image is converted to grayscale and thresholded into a binary mask. Dark blobs on a light background are the default; use `--foreground light` for light blobs on a dark background. Image or contour moments then give the centroid:

```text
center_x = m10 / m00
center_y = m01 / m00
```

Both implementations check `m00` before division. An empty mask or a zero-area contour therefore produces a clear error instead of invalid coordinates. Multiple-blob results are sorted by geometric position rather than relying on the unspecified order returned by `findContours`.

## Python setup and examples

From the repository root:

```bash
cd CenterofBlob
python3 -m pip install -r requirements.txt
```

Run either example without an input argument to use its bundled image:

```bash
python3 single_blob.py
python3 center_of_multiple_blob.py
```

The default commands open a GUI window. A headless validation run can also write one deterministic output:

```bash
python3 single_blob.py \
  --no-display --validate --output-dir output/single
python3 center_of_multiple_blob.py \
  --no-display --validate --output-dir output/multiple
```

The generated files are:

```text
output/
├── multiple/
│   └── multiple_blob_result.png
└── single/
    └── single_blob_result.png
```

Use `--input` to process another image:

```bash
python3 single_blob.py --input /path/to/image.png --no-display
python3 center_of_multiple_blob.py \
  --input /path/to/image.png --min-area 100 --no-display
```

`-i` and the article's older `--ipimage` spelling remain aliases for `--input`. `--validate` is intentionally limited to the bundled image because its assertions use known sample metrics.

## C++ build and examples

Build all three shipped entry points:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Pass `OpenCV_DIR` when multiple installations are present. For example:

```bash
cmake -S . -B build-opencv5 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv-5.0.0/lib/cmake/opencv5
cmake --build build-opencv5 --parallel
```

The CMake build rejects OpenCV versions outside major versions 4 and 5 and compiles with strict warnings (`-Wall -Wextra -Wpedantic -Werror`, or `/W4 /WX` with MSVC).

Like the Python programs, the C++ executables find their bundled inputs without depending on the current working directory:

```bash
./build/single_blob
./build/center_of_multiple_blob
```

Headless validation and output use the same options:

```bash
./build/single_blob \
  --no-display --validate --output-dir output/cpp-single
./build/center_of_multiple_blob \
  --no-display --validate --output-dir output/cpp-multiple
```

The original tutorial published the misspelled filename `center_of_multipe_blob.cpp`. It remains a supported compatibility entry point and is compiled and tested as `center_of_multipe_blob`; new commands and links should use `center_of_multiple_blob`.

## Command-line options

| Option | Meaning |
| --- | --- |
| `--input PATH`, `-i PATH`, `--ipimage PATH` | Override the bundled input |
| `--threshold 0..255` | Set the grayscale threshold; default `127` |
| `--foreground dark\|light` | Select the blob polarity; default `dark` |
| `--min-area N` | Ignore smaller contours; multiple-blob example only |
| `--output FILE` | Write the annotation to an exact filename |
| `--output-dir DIR` | Write the example's deterministic filename in `DIR` |
| `--no-display` | Suppress all GUI windows |
| `--validate` | Check the bundled asset and print `VALIDATION PASSED` |

`--output` and `--output-dir` are mutually exclusive. Output parent directories are created automatically.

## Regression results

The bundled files produce these stable values in Python and C++ with both tested OpenCV versions:

- `circle.png`: `225x225`, centroid `(112, 112)`, 35,272 foreground pixels.
- `multiple-blob.png`: `1089x236`, 70,791 foreground pixels.
- Six left-to-right centroids: `(72, 130)`, `(256, 115)`, `(439, 115)`, `(625, 115)`, `(807, 122)`, `(993, 116)`.
- Corresponding contour areas: `10194.0`, `13946.0`, `14583.0`, `16105.0`, `3688.0`, `14853.0`.

OpenCV 5 changed Hershey text rasterization, so annotated PNG bytes and a small number of label pixels are not expected to match OpenCV 4.14. In the macOS acceptance run, all source metrics above matched exactly; decoded OpenCV 4.14/5 annotations differed in at most 1.1% of pixels for the single image and 0.14% for the multiple image, with mean absolute channel differences no greater than 1.27 and 0.12 respectively. Python and C++ outputs were byte-for-byte identical within each OpenCV version. Cross-version checks should therefore compare the exact geometry and use these documented image tolerances instead of comparing encoded bytes.

## Tests

Run the 17 Python unit and real-CLI tests from this directory:

```bash
python3 -m unittest discover -s . -p "test_blob_centroids.py" -v
```

The CLI tests launch both scripts from an unrelated temporary directory, validate the real entry points, inspect generated images, assert the exact output set, and exercise invalid arguments and missing inputs.

Run the three C++ regression tests after building:

```bash
ctest --test-dir build --output-on-failure
```

CTest runs the canonical single- and multiple-blob executables plus the
misspelled compatibility executable. Each test uses its own build-local output
directory, removes only its known prior output, requires an explicit validation
marker, reloads the generated image, and permits exactly one expected file.

## Directory structure

```text
CenterofBlob/
├── cmake/
│   └── RunExampleTest.cmake
├── CMakeLists.txt
├── README.md
├── center_of_multipe_blob.cpp
├── center_of_multiple_blob.cpp
├── center_of_multiple_blob.py
├── circle.png
├── multiple-blob.png
├── requirements.txt
├── single_blob.cpp
├── single_blob.py
└── test_blob_centroids.py
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

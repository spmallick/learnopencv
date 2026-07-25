# Delaunay Triangulation and Voronoi Diagrams with OpenCV

This folder contains the Python and C++ examples for the LearnOpenCV article
[Delaunay Triangulation and Voronoi Diagram using OpenCV (C++/Python)](https://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/).
The examples insert facial landmarks into `Subdiv2D`, render the Delaunay
triangulation, and render its dual Voronoi diagram.

<p align="center"><img src="https://learnopencv.com/wp-content/uploads/2015/11/opencv-delaunay-vornoi-subdiv-example.jpg" alt="Delaunay triangulation and Voronoi diagram"></p>

## OpenCV 4.14 and OpenCV 5 compatibility

The complete Python and C++ acceptance matrix was run with:

| Implementation | OpenCV | Other tested tools |
| --- | --- | --- |
| Python | 4.14.0 and 5.0.0 | Python 3.14.3, NumPy 2.4.2 |
| C++ | 4.14.0 and 5.0.0 | Apple Clang 21.0.0, CMake 3.29.3, C++17 |

The Python implementation requires Python 3.10 or newer. Run the commands below
from the tracked `Delaunay/` directory unless the command uses absolute paths.

OpenCV 5 moved `cv::Subdiv2D` from `imgproc` into the new `geometry` module.
The C++ source includes and links that module only for OpenCV 5. The Python API
remains `cv2.Subdiv2D` in both supported major versions.

The upgraded examples also:

- resolve bundled inputs from the source directory, independent of the current
  working directory;
- replace the removed `CV_AA`, `CV_FILLED`, `cv2.cv`, `xrange`, and `np.int`
  interfaces;
- round floating-point `Subdiv2D` geometry explicitly before raster drawing;
- de-duplicate the two repeated records in `obama.txt` before insertion;
- canonicalize triangles and sort Voronoi cells by site before assigning a
  deterministic palette;
- generate repeatable `delaunay.png` and `voronoi.png` outputs;
- support headless execution, custom inputs, and stable geometry validation.

## Python

Enter the project directory:

```bash
cd Delaunay
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the bundled example headlessly and validate it:

```bash
python3 delaunay.py \
  --no-display \
  --no-animation \
  --validate \
  --output-dir output
```

To watch points being inserted and view the final diagrams:

```bash
python3 delaunay.py --display --animate --output-dir output
```

Run all Python regression tests:

```bash
python3 -m unittest discover -s tests -v
```

The tests execute the real CLI from an unrelated temporary directory, verify
both generated PNG files, repeat the run to check deterministic output, and
exercise missing, invalid, and output-colliding inputs.

## C++

OpenCV 5 requires a C++17 compiler. Configure a fresh build by pointing
`OpenCV_DIR` at the exact installation you want to test. GCC users should use
GCC 9 or newer so `std::filesystem` links without the legacy GCC 8
`stdc++fs` workaround.

OpenCV 4.14 example:

```bash
cmake -S . -B build-4.14 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv-4.14/lib/cmake/opencv4
cmake --build build-4.14 --parallel
ctest --test-dir build-4.14 --output-on-failure
```

OpenCV 5 example:

```bash
cmake -S . -B build-5.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/path/to/opencv-5.0/lib/cmake/opencv5
cmake --build build-5.0 --parallel
ctest --test-dir build-5.0 --output-on-failure
```

The executable embeds the source directory only to locate the bundled default
inputs. Custom inputs and outputs remain available:

```bash
./build-5.0/delaunay \
  --image /path/to/image.jpg \
  --points /path/to/points.txt \
  --output-dir /tmp/delaunay-output \
  --no-display
```

## Command-line options

Both implementations expose equivalent controls:

| Option | Purpose |
| --- | --- |
| `--image PATH` | Override the bundled `obama.jpg` image. |
| `--points PATH` | Override the bundled two-column landmark file. |
| `--output-dir PATH` | Select the directory for both PNG outputs. |
| `--display` / `--no-display` | Enable or disable final GUI windows. |
| `--animate` / `--no-animation` | Enable or disable insertion animation. Animation requires display. |
| `--animation-delay-ms N` | Set the positive delay between animated insertions. |
| `--validate` | Check the bundled-data regression contract. |

The point file accepts one finite integer `x y` pixel coordinate pair per line.
Blank lines and trailing `#` comments are allowed. Every point must lie inside
the image's half-open rectangle: `0 <= x < width` and
`0 <= y < height`. The CLI also rejects an output directory that would replace
the input image or landmark file.

## Outputs and validation

Each successful run writes:

- `delaunay.png`: the source image with triangulation edges and landmark sites;
- `voronoi.png`: deterministically colored Voronoi cells and their sites.

For the bundled 512×697 image, validation confirms:

- 68 landmark records, 66 unique sites, and 2 duplicates;
- 110 triangles, 175 unique edges, and 20 convex-hull edges;
- 66 nonvirtual Voronoi facets;
- a total triangulated area of 30,853 square pixels;
- matching Delaunay topology and Voronoi center sets;
- a Delaunay render that differs from the source and a nonblank,
  multicolor Voronoi render.

The tests compare ordering-independent geometric invariants because OpenCV does
not guarantee triangle or facet collection order. They do not rely on a
platform-specific hash of antialiased pixels.

## Project layout

```text
Delaunay/
├── .gitignore
├── CMakeLists.txt
├── README.md
├── delaunay.cpp
├── delaunay.py
├── obama.jpg
├── obama.txt
├── requirements.txt
└── tests/
    ├── check_cpp_failures.cmake
    ├── check_cpp_regression.cmake
    └── test_delaunay.py
```

`Subdiv2D` produces an ordinary Delaunay triangulation. It does not implement a
constrained Delaunay triangulation with user-enforced edges.


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

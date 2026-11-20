# [Robust Image Alignment with ALIKED, LightGlue, and MAGSAC++](https://learnopencv.com/robust-image-alignment/)

[<img src="figures/robust-image-alignment-thumbnail.webp" alt="Robust image alignment: real viewpoint change and the aligned overlay, using ALIKED, LightGlue, and MAGSAC++." width="900">](https://learnopencv.com/robust-image-alignment/)

This companion contains three runnable versions: the [complete notebook](robust_image_alignment.ipynb), a [pure Python module and CLI](robust_image_alignment.py), and a [native C++17 program](cpp/main.cpp). They cover natural photographs, planar documents and printed targets, including identification of a known appearance from one reference image. The [article source](article.md), explanatory diagrams, and recorded experiments accompany the code.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/robust-image-alignment-opencv-5.1.0-2026.11.20/Robust-Image-Alignment.zip)

The [versioned standalone release](https://github.com/spmallick/learnopencv/releases/tag/robust-image-alignment-opencv-5.1.0-2026.11.20) contains the tested `Robust-Image-Alignment/` folder. Download its [SHA-256 checksum](https://github.com/spmallick/learnopencv/releases/download/robust-image-alignment-opencv-5.1.0-2026.11.20/Robust-Image-Alignment.zip.sha256) alongside the ZIP. The release notes identify the source commit, tested platforms, and verification results. Sample inputs are fetched separately by the checked downloaders.

## Tested release and environment

All three versions target **OpenCV 5.1.0**. The [official November 6 release](https://github.com/opencv/opencv/releases/tag/5.1.0) resolves to commit `e4f78fbcf1a773fbed69e14500e8b734963eb585`. The reference build uses that source commit. PyPI also offers [opencv-python 5.1.0.95](https://pypi.org/project/opencv-python/5.1.0.95/). Install a single provider of the `cv2` namespace per environment.

The November 14 CUDA benchmark and November 20 rerun used Typhoon, Ubuntu 24.04, Intel Core i7-6850K, and NVIDIA GTX 1080 Ti. Python 3.14.3, NumPy 2.4.2, PyTorch 2.10.0+cu126, torchvision 0.25.0+cu126, and Kornia 0.8.2 were used. OpenCV geometry uses one CPU thread; PyTorch is configured for four CPU threads. The neural run uses one GPU. The November 20 rerun reproduced the real and synthetic accuracy results. We retained the November 14 timings because another CPU-intensive job was active on November 20. The ECC outcomes use the corrected initialization on November 20. See `results/experiments.json` and `results/opencv_build.txt` for actual runtime evidence. The notebook's first cell checks the imported OpenCV version.

## Install

Run these commands in this project directory. Python 3.14.3 was used for the recorded experiments. Linux prerequisites are Git, CMake 3.16+, a C++17 compiler, and matching Python development headers. The build also needs JPEG/PNG development libraries (for example Ubuntu packages `build-essential cmake git libjpeg-dev libpng-dev`; Python installations must supply their own matching headers and library). Source compilation requires several GB of free disk space.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Linux with the recorded GTX 1080 Ti, install the Pascal-compatible CUDA 12.6 builds before the requirements. On macOS or CPU-only systems, skip this step.

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

Then install the dependencies and build the OpenCV runtime in that environment.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-lightglue.txt
PYTHON_BIN="$VIRTUAL_ENV/bin/python" BUILD_JOBS=6 bash scripts/build_opencv.sh
python download_assets.py
python download_models.py
python -c 'import cv2; print(cv2.__version__, cv2.__file__); assert cv2.__version__ == "5.1.0"'
```

The `--no-deps` LightGlue install keeps its dependency resolver from installing a second package that supplies `cv2`. The OpenCV build handles geometry and image processing. The notebook uses Matplotlib for visualization.

Our own image loading, encoding, dimension checks, and thumbnail drawing use OpenCV. Matplotlib (notebook/figure plots) and Torchvision (imported by upstream LightGlue) still require Pillow internally. `requirements.txt` pins the patched Pillow 13.0.0 release for those indirect dependencies, keeping the version visible to GitHub security monitoring; removing our Pillow imports alone would not secure an older installed copy. For an existing environment, rerun `python -m pip install -r requirements.txt` before continuing. The original experiment records retain their historical package versions.

`OPENCV_BUILD_ROOT`, `OPENCV_INSTALL_ROOT`, `PYTHON_BIN`, and `BUILD_JOBS` customize the source build. The script verifies the exact release commit and runtime version. It requires network access for source/dependency downloads; asset and weight downloads also need network access on the first run.

## Run the notebook

```bash
python -m ipykernel install --user --name robust-alignment-5-1 --display-name "Robust alignment OpenCV 5.1"
python -m jupyterlab robust_image_alignment.ipynb
```

Select the kernel installed from this environment and run all cells. Start Jupyter from this directory or the repository root. The notebook runs the complete demonstrations, estimates geometry directly, evaluates known homographies, tests absent targets, compares estimators, exercises optional ECC, and includes an exercise with a working answer. Its optional full-benchmark flag is off by default; recorded results are explicitly identified as stored evidence rather than timings from the current kernel.

## Align custom images

```bash
python robust_image_alignment.py \
  --reference assets/downloads/opencv/box.png \
  --moving assets/downloads/opencv/box_in_scene.png \
  --method lightglue --estimator magsac --device cpu \
  --output-dir alignment-output/box --no-display --validate
```

Use `--method sift` for SIFT plus a ratio test, `--method aliked-nn` for ALIKED plus a ratio test, or `--method lightglue` for ALIKED plus LightGlue. The estimator is `ransac` or `magsac`. Device selection is `auto`, `cpu`, `cuda`, or `mps`; CUDA and MPS must actually be available. The article’s main neural benchmark uses CUDA. The eight-pair backend comparison checks the same acceptance and 3 px error criterion on CPU, CUDA, and Apple MPS separately.

`--max-keypoints` defaults to 2048; `--resize` caps the longest image side at 1024 without upsampling (0 disables resizing). `--threshold` is in original reference-image pixels, and `--min-inliers` defaults to 12. All returned homographies map **moving → reference**. Invert that transform to locate reference corners in the scene. For resized SIFT keypoints, both languages map each axis back to original pixel centers as `(coordinate + 0.5) / resize_factor - 0.5`.

Accepted CLI results contain `report.json`, `matches.jpg`, `aligned.png`, `valid_mask.png`, `overlay.jpg`, and `reference_boundary.jpg`. Rejected results contain `report.json` and `matches.jpg`; a rejected rerun removes only the four owned acceptance images while preserving unrelated files. `--validate` exits 3 for a rejected alignment; invalid arguments/inputs exit 2. The saved mask tests nearest-pixel support. For differences or other intensity measurements, pass `full_support=True` to Python’s `warp_alignment` to require every nonzero bilinear interpolation weight to come from inside the source image. The C++ helper exposes the same option after `require_accepted`. The program is headless. Scripts and bundled input download paths resolve independently of the caller's current directory; explicit input/output arguments are relative to that caller unless absolute paths are supplied.

## Reproduce the evidence

```bash
python -m unittest discover -s tests -v
python scripts/check_article.py
python scripts/check_clis.py
python run_experiments.py --device cuda --repeats 5 --study-seeds 12
python run_experiments.py --device cpu --check-backend
python run_experiments.py --device cuda --check-backend
# On an Apple host, use --device mps --check-backend for the MPS comparison.
python make_figures.py --device cuda
python download_assets.py --verify-only
python scripts/render_article.py
```

The real-image comparison uses two Oxford sequences, with image 1→2 in each sequence reserved for validation and image 1→3…6 as eight evaluation pairs. Repeated timing trials are not additional independent image pairs. The criterion is an accepted estimate with median ground-truth overlap-grid error below three original reference pixels. This is a compact teaching comparison, not a broad benchmark or a production false-acceptance evaluation.

The controlled estimator study has 160 synthetic correspondences, 60% uniform outliers, three noise levels, three thresholds, and 12 deterministic data seeds. The two OpenCV estimator presets receive identical correspondences for each seed. The sweep varies both the estimator threshold and the median inlier residual gate. It does not model coherent repeated-structure errors. ECC is evaluated separately. Both implementations convert the initial moving-to-reference matrix into the reference-to-moving convention and normalize the inverse to a bottom-right entry of one before calling ECC. The returned candidate is converted back to moving-to-reference.

Full pair timing includes both image extractions, matching, geometry/acceptance, and accepted image/mask warping. It excludes disk I/O, model loading, warm-up, plotting, and the optional ECC stage. Reference caching changes the workload; do not treat matcher FPS as end-to-end throughput. Runtime distributions and their scope are recorded with the results.

## Build and run the C++ version

The C++ executable performs feature extraction, matching, robust geometry, rejection checks, warping, and diagnostics natively. ALIKED-n16 and LightGlue use ONNX Runtime's CPU backend with the notebook's original weights. Python is not needed to configure, build, download sample images, or run it. Eleven bundled ONNX graphs (~50.8 MB) retain ALIKED's detector and LightGlue's adaptive stopping/pruning; CMake verifies their SHA-256 hashes. See [model provenance and export notes](cpp/models/NOTICE.md).

Use CMake 3.18+, Git, a C++17 compiler, and JPEG/PNG development libraries. These commands start in this project directory. The source-build helper installs the exact released OpenCV modules needed by C++:

```bash
OPENCV_CPP_BUILD_ROOT="$PWD/.deps/opencv-5.1.0" BUILD_JOBS=6 bash scripts/build_opencv_cpp.sh
cmake -DASSETS_DIR="$PWD/assets/downloads" -P cpp/download_assets.cmake
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$PWD/.deps/opencv-5.1.0/install/lib/cmake/opencv5"
cmake --build build-cpp --parallel 6
ctest --test-dir build-cpp --output-on-failure
./build-cpp/robust_image_alignment_cpp \
  --reference assets/downloads/opencv/box.png \
  --moving assets/downloads/opencv/box_in_scene.png \
  --method lightglue --estimator magsac --device cpu \
  --output-dir alignment-output-cpp/box --no-display --validate
```

An existing OpenCV 5.1.0 installation can be selected through `OpenCV_DIR` instead. The helper's `OPENCV_SOURCE_DIR`, `OPENCV_CPP_INSTALL_ROOT`, and `BUILD_JOBS` options customize the build. This companion intentionally targets 5.1.0: OpenCV 5 moved SIFT and homography headers/modules, so it does not claim OpenCV 4 compatibility.

Configuration downloads Microsoft's pinned ONNX Runtime 1.24.3 NuGet archive (~125 MB), verifies its hash, and extracts the current platform's native CPU library and headers. Automatic SDK setup supports Linux x86-64 and macOS arm64/x86-64; Linux x86-64 and macOS arm64 were tested. To use an existing SDK, pass `-DONNXRUNTIME_ROOT=/absolute/path` with `include/` and `lib/` subdirectories. Other platforms require such an installation and are untested. `-DBUILD_ALIGNMENT_APP=OFF` builds only geometry and its tests without ONNX Runtime.

The pair CLI has the same three matching routes (`sift`, `aliked-nn`, `lightglue`), two estimators (`ransac`, `magsac`), feature/resize budgets, acceptance thresholds, six normal output files, and exit-code contract as Python. `--refine-ecc` additionally saves a refinement candidate and mask without replacing the original estimate. `--dump-features` saves keypoints, descriptors, and paired correspondences for inspection. All homographies map moving → reference, and valid black pixels are retained through an independently warped mask.

The following native workflows reproduce the notebook's operations:

```bash
./build-cpp/robust_image_alignment_cpp --demo-all \
  --output-dir alignment-output-cpp/demos --no-display --validate
./build-cpp/robust_image_alignment_cpp --benchmark --repeats 5 --study-seeds 12 \
  --output-dir alignment-output-cpp/benchmark --no-display --validate
./build-cpp/robust_image_alignment_cpp --synthetic-study --study-seeds 12 \
  --output-dir alignment-output-cpp/synthetic --no-display --validate
```

`--demo-all` covers perspective and illumination changes, document alignment, single-reference box identification, absent targets, a cached three-reference catalogue with ambiguity handling, fixed-correspondence RANSAC/MAGSAC++ comparison, optional ECC, a parallax diagnostic, and the spatial-support exercise. It also runs the controlled synthetic study. `--benchmark` evaluates the same ten Oxford pairs and all six matcher/estimator combinations, retaining failed estimates and reporting common-success timing distributions. It writes `benchmark_report.json` and `benchmark.csv`; demos and synthetic runs write `demo_report.json` and `synthetic_study.json`.

Default asset/model paths resolve from the compiled project directory, independently of the current working directory. After relocating the project, pass `--project-dir /absolute/path/to/Robust-Image-Alignment`; `--assets-dir` and `--model-dir` can override individual locations. Custom input and output paths are relative to the caller unless absolute paths are supplied. Every mode is headless.

C++ uses CPU inference, whereas the article's stored neural timing results use PyTorch on a GTX 1080 Ti. CPU and GPU LightGlue use different upstream pruning thresholds, and floating-point differences can change near-threshold decisions. C++ results therefore have their own evidence and are not substituted for the article's GPU measurements. The synthetic C++ study uses `std::mt19937`; Python uses NumPy's generator. The distributions and protocol agree, while individual draws differ. Shared point fixtures test geometric equivalence independently of these random streams.

With default pruning, Python CPU and Apple MPS each passed 7/8, and Python CUDA passed 8/8 on the two Oxford sequences. On `graf_1_6`, the Python CPU, MPS and native C++ CPU estimates passed the geometric gates with a median grid error of 11.97 px. CUDA gave 1.76 px. Switching off point pruning, with the same cached features on each device, also gave 1.76 px on CPU and MPS for this pair. See `results/backend_cpu.json`, `results/backend_cuda.json` and `results/backend_mps.json` for the per-pair checks. Like the notebook, demo validation checks acceptance and reports independent accuracy separately through `benchmark_success_3px`; the benchmark retains its strict accuracy requirement.

The optional Python development tool `scripts/check_cpp_parity.py` compares native outputs with the original Python CPU implementation using shared geometry fixtures and real image pairs. Normal native use does not require this tool. Model re-export is also optional; checked-in graphs are ready to run.

The native suite includes 11 CTest checks, with 16 geometry cases and a SIFT pixel-center regression. The Python unit suite includes 17 cases. The Python comparison passes five shared geometry cases and twelve real pipelines. Each native benchmark contains 180 real-image trials and 216 synthetic trials. See [retained C++ validation evidence](results/cpp/README.md) for the exact scope, outcomes, and CPU accuracy caveat. To rerun the optional comparison in the pinned Python environment:

```bash
python scripts/check_cpp_parity.py \
  --executable "$PWD/build-cpp/robust_image_alignment_cpp" \
  --methods sift aliked-nn lightglue --model-dir "$PWD/cpp/models" \
  --output alignment-output-cpp/python-parity.json
```

## Inputs and rights

`download_assets.py` downloads 35 files (~14.8 MB) using immutable URLs and verifies sizes and SHA-256 hashes. `assets_manifest.json` records provenance, ground-truth direction, and validation/evaluation splits. Images remain under their original source terms. Oxford capture sequences are download-only; repository code licences are not presented as blanket image rights. See [asset notes](research/sources-assets.md) and the retained OpenCV notices. Model hashes and exact download sources are in `download_models.py`.

## Project files

```text
Robust-Image-Alignment/
├── README.md / .gitignore
├── article.md / article.html
├── robust_image_alignment.ipynb
├── robust_image_alignment.py
├── CMakeLists.txt
├── cpp/
│   ├── main.cpp / workflows.cpp / workflows.hpp
│   ├── alignment.cpp / alignment.hpp
│   ├── feature_matcher.cpp / feature_matcher.hpp
│   ├── download_assets.cmake / onnxruntime.cmake / verify_models.cmake
│   ├── models/         # Eleven ONNX graphs, hashes, validation, notices
│   └── tests/          # Native geometry and CLI regressions
├── download_assets.py / download_models.py
├── assets_manifest.json
├── requirements.txt / requirements-lightglue.txt
├── run_experiments.py / make_figures.py / build_notebook.py
├── scripts/            # Python/C++ source builds, export, parity, article checks
├── tests/              # Geometry, rejection, and CLI regressions
├── figures/            # Diagrams and measured photographic results
├── results/            # Raw experiments and environment evidence
├── research/           # Primary sources and provenance
├── publication/        # SEO, release, and newsletter publication records
└── assets/             # Retained notices; downloaded pixels are ignored
```

## Related LearnOpenCV articles

- [Feature Based Image Alignment using OpenCV](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
- [Image Alignment using ECC](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
- [Homography Examples using OpenCV](https://learnopencv.com/homography-examples-using-opencv-python-c/)
- [Automatic Document Scanner using OpenCV](https://learnopencv.com/automatic-document-scanner-using-opencv/)
- [Introduction to Feature Matching Using Neural Networks](https://learnopencv.com/feature-matching/)

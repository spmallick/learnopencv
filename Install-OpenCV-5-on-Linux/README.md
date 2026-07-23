# Install OpenCV 5 on Linux: Verified Companion for OpenCV 5.0.0

This directory is the GitHub companion for LearnOpenCV's [**Install OpenCV 5 on Linux**](https://learnopencv.com/install-opencv-5-linux/) guide. It provides two narrow tools:

- `verify-opencv5.sh` checks a completed wheel or source installation and writes an evidence directory.
- `configure-opencv5.sh` reproduces the guide's tested `desktop-cpu` or `headless-cuda` CMake configuration. It prints the command by default and configures only when `--run` is supplied.

The bundle does not install packages, clone repositories, build OpenCV, install OpenCV, change services, or reboot the host. The verifier does not use the network or alter the selected Python environment or OpenCV prefix. A `--run` configuration can trigger downloads performed by OpenCV's own CMake logic, as explained in the article.

## Requirements

- Bash 4 or newer, Git, GNU `realpath`, CMake 3.18 or newer, and Ninja
- Python 3 with the completed OpenCV installation
- A C++17 compiler for source-install verification or configuration
- `ldd` and `pkg-config` for source-install verification
- An active graphical display or Xvfb for non-headless wheel verification; Xvfb for the `desktop-cpu` runtime test
- `nvidia-smi` on `PATH` (or WSL's `/usr/lib/wsl/lib/nvidia-smi`), a working NVIDIA driver, CUDA Toolkit, and GPU for the tested x86-64 `headless-cuda` verifier

All path arguments must be absolute. The configure helper rejects unsafe or existing destination prefixes, dirty source trees, and existing build directories. The source verifier requires an existing installed prefix and rejects an existing evidence directory. Both scripts reject unknown arguments.

## Verify a Python wheel

Install exactly one wheel in its own virtual environment first. Then run:

```bash
./verify-opencv5.sh wheel \
  --python "$HOME/.venvs/opencv5-wheel/bin/python" \
  --variant opencv-contrib-python-headless \
  --version 5.0.0 \
  --evidence "$PWD/evidence-wheel"
```

Allowed variants are:

- `opencv-python`
- `opencv-contrib-python`
- `opencv-python-headless`
- `opencv-contrib-python-headless`

The test requires the imported module to live inside the selected Python environment, confirms that exactly one wheel family and the exact 5.0.0.93 revision are installed, performs image processing and in-memory PNG I/O, checks contrib expectations, and requires zero CUDA devices. Non-headless variants also open a real HighGUI window on the active display or under Xvfb.

## Verify a source installation

Desktop CPU profile:

```bash
./verify-opencv5.sh source \
  --profile desktop-cpu \
  --prefix "$HOME/.local/opencv/5.0.0" \
  --python "$HOME/.venvs/opencv-5.0.0/bin/python" \
  --version 5.0.0 \
  --evidence "$PWD/evidence-cpu"
```

Headless CUDA profile:

```bash
./verify-opencv5.sh source \
  --profile headless-cuda \
  --prefix "$HOME/.local/opencv/5.0.0-cuda12.9" \
  --python "$HOME/.venvs/opencv-5.0.0-cuda12.9/bin/python" \
  --cuda-root /usr/local/cuda-12.9 \
  --dnn-cuda on \
  --version 5.0.0 \
  --evidence "$PWD/evidence-cuda"
```

Source verification adds an exact-version C++17 test, checks that every linked OpenCV library resolves under the selected prefix, and requires `opencv5.pc` to report the requested version. The CPU profile executes FFmpeg, GStreamer, and GTK3 under Xvfb. The CUDA profile performs an upload, CUDA color-conversion kernel, and download in both Python and C++ on every visible GPU. Set `--dnn-cuda on` when the build contract includes cuDNN and the classic DNN CUDA backend; use `off` only when both were explicitly disabled at configuration time.

Both source verification modes require an installed Python binding. For a C++-only build, use the article's native CMake, runtime, `ldd`, and pkg-config checks instead.

Optional `--source` and `--contrib` paths record Git revisions and require the exact OpenCV 5.0.0 commits used by the guide.

The evidence path must not exist. A successful run leaves all reports and the C++ test build there. A failed run also preserves partial evidence for diagnosis. The script never removes or overwrites that directory.

## Print or run a tested CMake configuration

The helper supports only the two contracts tested by the article. It requires fresh source clones, an absent build directory, a safe versioned prefix, and a Python environment that already contains NumPy.

Desktop CPU dry run:

```bash
./configure-opencv5.sh desktop-cpu \
  --source "$HOME/src/opencv-5.0.0/opencv" \
  --contrib "$HOME/src/opencv-5.0.0/opencv_contrib" \
  --build "$HOME/src/opencv-5.0.0/build-cpu-new" \
  --prefix "$HOME/.local/opencv/5.0.0-new" \
  --python "$HOME/.venvs/opencv-5.0.0/bin/python" \
  --version 5.0.0
```

Headless CUDA dry run:

```bash
./configure-opencv5.sh headless-cuda \
  --source "$HOME/src/opencv-5.0.0/opencv" \
  --contrib "$HOME/src/opencv-5.0.0/opencv_contrib" \
  --build "$HOME/src/opencv-5.0.0/build-cuda-new" \
  --prefix "$HOME/.local/opencv/5.0.0-cuda-new" \
  --python "$HOME/.venvs/opencv-5.0.0-cuda12.9/bin/python" \
  --cuda-root /usr/local/cuda-12.9 \
  --cuda-arch-cmake 61 \
  --cuda-arch-opencv 6.1 \
  --dnn-cuda on \
  --version 5.0.0
```

Inspect the printed command. Add `--run` only when the paths, dependencies, CUDA architecture, and requested backends are correct. The helper stops after CMake configuration; build and install remain explicit steps from the article.

The configure helper intentionally accepts one CUDA compute capability per build and requires the compact and dotted forms to agree—for example, `61` and `6.1`. Use a separately reviewed CMake configuration for heterogeneous or redistributable multi-architecture binaries.

## Trust and portability boundary

The wheel verifier is designed around the official 5.0.0.93 wheel contracts. The source profiles reproduce and validate the exact Ubuntu 24.04 x86-64 builds documented in the article. Path discovery accepts both `lib` and `lib64`, but that alone does not make the Ubuntu dependency recipe portable to every distribution. The `headless-cuda` verifier requires a working `nvidia-smi` query and is not Jetson-compatible without adaptation; WSL's limited `nvidia-smi` may also require the article's manual checks. WSL2, Jetson, Alpine, ARM, and non-Ubuntu readers should first follow the article's platform router and then adapt only the relevant source profile.

## Download the standalone companion bundle

Download the immutable, versioned companion bundle and its checksum:

- [Install-OpenCV-5-on-Linux.zip](https://github.com/spmallick/learnopencv/releases/download/install-opencv-5-linux-2026.07.23/Install-OpenCV-5-on-Linux.zip)
- [Install-OpenCV-5-on-Linux.zip.sha256](https://github.com/spmallick/learnopencv/releases/download/install-opencv-5-linux-2026.07.23/Install-OpenCV-5-on-Linux.zip.sha256)

On Linux, download, verify, and extract it with:

```bash
release_url="https://github.com/spmallick/learnopencv/releases/download/install-opencv-5-linux-2026.07.23"

curl --fail --location --remote-name \
  "${release_url}/Install-OpenCV-5-on-Linux.zip"
curl --fail --location --remote-name \
  "${release_url}/Install-OpenCV-5-on-Linux.zip.sha256"

sha256sum --check Install-OpenCV-5-on-Linux.zip.sha256
unzip Install-OpenCV-5-on-Linux.zip
cd Install-OpenCV-5-on-Linux
```

The archive contains one top-level project folder and these six tracked files:

```text
Install-OpenCV-5-on-Linux/
├── README.md
├── configure-opencv5.sh
├── cpp-smoke/
│   ├── CMakeLists.txt
│   └── main.cpp
├── verify-opencv5.sh
└── verify_python.py
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

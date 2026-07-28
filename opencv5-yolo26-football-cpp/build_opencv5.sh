#!/usr/bin/env bash
# Build OpenCV 5.0.0 from source with the DNN module (CPU, new engine).
#
# There is no pip-style shortcut for the OpenCV 5 C++ development files yet, so
# the C++ demos need OpenCV built from source. Works on Linux/macOS with the
# native compiler, and on Windows with the MSYS2 MinGW-w64 (ucrt64) toolchain.
#
# Prereqs: git, cmake, ninja, a C++17 compiler, and the FFmpeg dev libraries
# (needed for MP4 read/write):
#   MSYS2 : pacman -S mingw-w64-ucrt-x86_64-{cmake,ninja,ffmpeg}
#   Debian: apt install cmake ninja-build libavcodec-dev libavformat-dev libswscale-dev
set -euo pipefail

SRC=opencv5_src
BUILD=opencv5_build
INSTALL="$(pwd)/opencv5_install"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch 5.0.0 https://github.com/opencv/opencv.git "$SRC"
else
  expected_commit=$(git -C "$SRC" rev-list -n 1 5.0.0)
  actual_commit=$(git -C "$SRC" rev-parse HEAD)
  if [ "$actual_commit" != "$expected_commit" ]; then
    echo "$SRC is not checked out at OpenCV 5.0.0" >&2
    exit 1
  fi
fi

cmake -G Ninja -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$INSTALL" \
  -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,dnn,video \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DWITH_CUDA=OFF -DWITH_FFMPEG=ON \
  "$SRC"

# Confirm the configuration summary prints "FFMPEG: YES" before the long compile.
cmake --build "$BUILD"
cmake --install "$BUILD"

echo
echo "OpenCV 5 installed to: $INSTALL"
echo "Now build the demos:"
echo "  cmake -G Ninja -B build -DOpenCV_DIR=$INSTALL/lib/cmake/opencv5   # Linux/macOS"
echo "  cmake -G Ninja -B build -DOpenCV_DIR=$INSTALL/x64/mingw/lib       # Windows/MinGW"
echo "  cmake --build build"

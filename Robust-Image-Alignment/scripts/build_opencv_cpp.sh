#!/usr/bin/env bash
# Build the released OpenCV 5.1.0 modules used by the native C++ companion.
set -euo pipefail
build_root="${OPENCV_CPP_BUILD_ROOT:-${TMPDIR:-/tmp}/robust-opencv-cpp-5.1.0}"
source_dir="${OPENCV_SOURCE_DIR:-${build_root}/source}"
install_dir="${OPENCV_CPP_INSTALL_ROOT:-${build_root}/install}"
release_commit="e4f78fbcf1a773fbed69e14500e8b734963eb585"
mkdir -p "$build_root"
if [[ ! -e "$source_dir/.git" ]]; then
    git clone --depth 1 --branch 5.1.0 https://github.com/opencv/opencv.git "$source_dir"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$release_commit" ]] || {
    echo "Source must be the verified OpenCV 5.1.0 release commit" >&2
    exit 1
}
# A matching commit alone does not prove that a reused source tree is unmodified.
if ! git -C "$source_dir" diff --quiet || ! git -C "$source_dir" diff --cached --quiet; then
    echo "OpenCV release source has tracked modifications; use a clean source checkout" >&2
    exit 1
fi
# Python bindings are excluded: the resulting example runs as a native program.
cmake -S "$source_dir" -B "$build_root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DBUILD_LIST=core,imgproc,imgcodecs,features,geometry,video \
    -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
    -DBUILD_opencv_dnn=OFF -DWITH_ONNXRUNTIME=OFF -DDOWNLOAD_ONNXRUNTIME=OFF \
    -DWITH_PROTOBUF=OFF -DBUILD_PROTOBUF=OFF \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=OFF \
    -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_OPENEXR=OFF -DWITH_VTK=OFF \
    -DWITH_QT=OFF -DWITH_GTK=OFF
cmake --build "$build_root/build" --parallel "${BUILD_JOBS:-6}"
cmake --install "$build_root/build"
"$install_dir/bin/opencv_version"
printf 'OpenCV_DIR=%s\n' "$install_dir/lib/cmake/opencv5"

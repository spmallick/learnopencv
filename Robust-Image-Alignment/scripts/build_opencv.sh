#!/usr/bin/env bash
# Build the latest stable OpenCV 5.x release verified for this article.
set -euo pipefail
python_bin="${PYTHON_BIN:-python3}"
work_dir="${OPENCV_BUILD_ROOT:-${TMPDIR:-/tmp}/robust-opencv-5.1.0}"
install_dir="${OPENCV_INSTALL_ROOT:-${work_dir}/install}"
source_dir="${work_dir}/source"
release_commit="e4f78fbcf1a773fbed69e14500e8b734963eb585"
mkdir -p "$work_dir"
if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 --branch 5.1.0 https://github.com/opencv/opencv.git "$source_dir"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$release_commit" ]] || {
    echo "OpenCV source is not the verified 5.1.0 release commit" >&2
    exit 1
}
python_include="$($python_bin -c 'import sysconfig; print(sysconfig.get_path("include"))')"
python_library="$($python_bin -c 'import os,sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')"
numpy_include="$($python_bin -c 'import numpy; print(numpy.get_include())')"
python_site="$($python_bin -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
# A minimal CPU build supplies geometry/warping; PyTorch handles neural inference.
cmake -S "$source_dir" -B "$work_dir/build" -U 'PYTHON3*' \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$install_dir" \
    -DBUILD_LIST=core,imgproc,imgcodecs,features,geometry,video,python3,python_bindings_generator \
    -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON -DBUILD_opencv_python_bindings_generator=ON \
    -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
    -DWITH_AVFOUNDATION=OFF -DWITH_OPENCL=OFF -DWITH_IPP=OFF \
    -DWITH_OPENEXR=OFF -DWITH_VTK=OFF -DWITH_QT=OFF -DWITH_GTK=OFF \
    -DPYTHON3_EXECUTABLE="$python_bin" -DPYTHON3_INCLUDE_DIR="$python_include" \
    -DPYTHON3_LIBRARY="$python_library" \
    -DPYTHON3_NUMPY_INCLUDE_DIRS="$numpy_include" \
    -DPYTHON3_PACKAGES_PATH="$python_site" -DOPENCV_PYTHON3_INSTALL_PATH="$python_site"
cmake --build "$work_dir/build" --parallel "${BUILD_JOBS:-6}"
cmake --install "$work_dir/build"
"$python_bin" -c 'import cv2; print("OpenCV runtime:", cv2.__version__, cv2.__file__); assert cv2.__version__ == "5.1.0"'

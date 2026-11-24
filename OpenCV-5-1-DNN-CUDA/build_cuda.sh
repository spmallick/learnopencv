#!/usr/bin/env bash
# Build and install the two inference engines into an isolated Linux venv.
set -euo pipefail
build_root="${BUILD_ROOT:-$PWD/.opencv51-cuda}"
mkdir -p "$build_root"
build_root="$(cd "$build_root" && pwd)"
cuda_root="${CUDA_ROOT:-/usr/local/cuda}"
export PATH="$cuda_root/bin:$PATH"
# Select the GPU's native instruction set, e.g. 61 for a GeForce GTX 1080 Ti.
cuda_arch="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR == 1 {gsub(/[.]/, ""); print}')}"
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || { echo 'Set CUDA_ARCH to the numeric compute capability (e.g. 86).' >&2; exit 1; }
nvcc --version

# A release tag is convenient for shallow clones. Verify its immutable commit, too.
clone_release() {
    local repo="$1" sha="$2"
    if [[ ! -d "$build_root/$repo/.git" ]]; then
        git clone --depth 1 --branch 5.1.0 "https://github.com/opencv/$repo.git" "$build_root/$repo"
    fi
    if [[ "$(git -C "$build_root/$repo" rev-parse HEAD)" != "$sha" || -n "$(git -C "$build_root/$repo" status --porcelain --untracked-files=no)" ]]; then
        echo "Expected a clean OpenCV 5.1.0 checkout in $build_root/$repo" >&2
        exit 1
    fi
}
clone_release opencv e4f78fbcf1a773fbed69e14500e8b734963eb585
clone_release opencv_contrib b67a855188a3d683813731010f6d39ac614733a4

python3 -m venv "$build_root/venv"
python="$build_root/venv/bin/python"
"$python" -m pip install 'numpy==2.2.6' 'onnxruntime-gpu==1.22.0' 'onnx==1.18.0'
# cuBLAS comes from the installed toolkit. Pin the cuDNN runtime and headers
# in the venv so ORT and OpenCV use the same Pascal-compatible build.
"$python" -m pip install --no-deps 'nvidia-cudnn-cu12==9.10.2.21'
site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cudnn="$site/nvidia/cudnn"
# OpenCV's linker expects the unversioned development-library name.
ln -sfn libcudnn.so.9 "$cudnn/lib/libcudnn.so"
# Activating this file makes cuDNN's dynamically loaded kernels available to
# both engines in fresh shells. Keep the user's existing library paths last.
printf 'source %q\nexport LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
    "$build_root/venv/bin/activate" "$cudnn/lib:$cuda_root/lib64" > "$build_root/activate.sh"
source "$build_root/activate.sh"
numpy_headers="$("$python" -c 'import numpy; print(numpy.get_include())')"
# cudev supplies the device headers required by OpenCV core. Limit the build to
# the Python binding, image I/O, preprocessing and DNN, plus their dependencies.
cmake -S "$build_root/opencv" -B "$build_root/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$build_root/install" \
    -DCMAKE_INSTALL_RPATH="$build_root/install/lib;$cudnn/lib" \
    -DCUDA_TOOLKIT_ROOT_DIR="$cuda_root" \
    -DCUDA_ARCH_BIN="$cuda_arch" -DCUDA_ARCH_PTX="" \
    -DOPENCV_EXTRA_MODULES_PATH="$build_root/opencv_contrib/modules" \
    -DCUDNN_LIBRARY="$cudnn/lib/libcudnn.so" -DCUDNN_INCLUDE_DIR="$cudnn/include" \
    -DBUILD_LIST=cudev,core,imgproc,imgcodecs,geometry,dnn,python3,python_bindings_generator \
    -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_JAVA=OFF -DBUILD_opencv_apps=OFF -DWITH_OPENCL=OFF \
    -DWITH_IPP=OFF -DWITH_OPENEXR=OFF -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_VTK=OFF \
    -DCPU_BASELINE=SSE3 -DCPU_DISPATCH=AVX2 \
    -DPYTHON3_EXECUTABLE="$python" -DPYTHON3_NUMPY_INCLUDE_DIRS="$numpy_headers" \
    -DPYTHON3_PACKAGES_PATH="$site" -DOPENCV_PYTHON3_INSTALL_PATH="$site"
cmake --build "$build_root/build" --parallel "${BUILD_JOBS:-6}"
cmake --install "$build_root/build"

# Device enumeration and backend registration are prerequisites. The benchmark
# checks each model's placement separately because operator support varies.
"$python" - <<'PY'
import ctypes

import cv2
import onnxruntime as ort

# 9.10.2 reports 91002 through the public cuDNN version API.
assert ctypes.CDLL("libcudnn.so.9").cudnnGetVersion() == 91002
assert cv2.__version__ == "5.1.0", cv2.__version__
assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "No CUDA device visible to OpenCV"
assert cv2.dnn.DNN_TARGET_CUDA in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)
assert "CUDAExecutionProvider" in ort.get_available_providers()
print(f"OpenCV {cv2.__version__}, ONNX Runtime {ort.__version__}, Python binding {cv2.__file__}")
PY
echo "Activate the environment with: source \"$build_root/activate.sh\""

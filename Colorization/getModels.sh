#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
MODEL_PATH="${MODEL_DIR}/colorization_eccv16.onnx"
MODEL_URL="https://github.com/spmallick/learnopencv/releases/download/colorization-opencv-2026.07.25/colorization_eccv16.onnx"
EXPECTED_SHA256="a1680679b609ca4d107edb83b8ac89c283cc474ce0a81edd6f01db85910e8201"
TEMP_PATH="${MODEL_PATH}.download"

mkdir -p "${MODEL_DIR}"
curl --fail --location --retry 3 "${MODEL_URL}" --output "${TEMP_PATH}"

ACTUAL_SHA256="$(shasum -a 256 "${TEMP_PATH}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    rm -f "${TEMP_PATH}"
    echo "Checksum mismatch for ${MODEL_PATH}" >&2
    exit 1
fi

mv "${TEMP_PATH}" "${MODEL_PATH}"
echo "Verified model: ${MODEL_PATH}"

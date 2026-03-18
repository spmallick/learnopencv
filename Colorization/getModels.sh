#!/usr/bin/env bash

set -euo pipefail

mkdir -p models

# The small support files are now committed in this repo. Only the large
# Caffe weights are downloaded on demand from GitHub release assets.
wget https://github.com/spmallick/learnopencv/releases/download/Colorization/colorization_release_v2.caffemodel -O ./models/colorization_release_v2.caffemodel

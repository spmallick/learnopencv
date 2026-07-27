# Model License and Provenance

## Current primary model

- Model: `face_detection_yunet_2026may.onnx`
- Upstream project: OpenCV Zoo
- Upstream path: `models/face_detection_yunet/face_detection_yunet_2026may.onnx`
- Pinned upstream commit: `47534e27c9851bb1128ccc0102f1145e27f23f98`
- Download URL: `https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2026may.onnx`
- Size: `229,738` bytes
- SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`
- License: MIT License, copyright Shiqi Yu
- Model family: YuNet

This dynamic-input model is the OpenCV Zoo default for variable-size inference
with OpenCV 5's ONNX Runtime engine. `download_models.py` verifies the pinned Git
LFS object before placing it under `models/`; the ONNX file is not committed.

### YuNet MIT license

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Historical assets

- `models/haarcascade_frontalface_default.xml` is retained for the optional
  OpenCV Haar baseline. Haar availability depends on the OpenCV build and is not
  part of the OpenCV 5 default test path.
- `models/mmod_human_face_detector.dat` and the MMOD examples are retained as
  legacy tutorial material only.
- The Caffe and TensorFlow SSD files under `models/` are retained as historical
  assets for the original 2018 article. The current programs do not load them,
  and Caffe is not part of the build or test path.
- `dlib.zip` is retained as a historical snapshot. Current optional dlib HOG
  builds use an installed dlib package rather than unzipping this archive.

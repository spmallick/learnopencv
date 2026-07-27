# Model License and Provenance

## Current runnable model

- Model: `pose_estimation_mediapipe_2023mar.onnx`
- Upstream project: OpenCV Zoo
- Upstream path: `models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx`
- Pinned upstream commit: `47534e27c9851bb1128ccc0102f1145e27f23f98`
- Download URL: `https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx`
- Size: `5,557,238` bytes
- SHA-256: `9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`
- License: Apache License 2.0, as recorded in the pinned OpenCV Zoo model directory
- Original model family: MediaPipe Pose / BlazePose

The model is downloaded on demand and is not committed to this repository.
`download_models.py` verifies both its size and SHA-256 digest before placing it
under `models/`.

## Legacy CMU OpenPose material

The original 2018 tutorial used CMU OpenPose Caffe models. Those weights are not
distributed, downloaded, converted, tested, or loaded by the current examples.
CMU's OpenPose model terms are not equivalent to the Apache-2.0 terms above and
include non-commercial restrictions. The old prototxt files and notebook remain
only as clearly identified historical tutorial material; they are not part of
the current CMake targets, Python entry points, tests, or model downloader.

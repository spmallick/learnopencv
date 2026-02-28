# FaceBlurPixelate

## 1. Python: How to Use

### What this does
This Python app opens a webcam feed, detects faces using OpenCV YuNet (neural network), and anonymizes each face using either:
- `blur`
- `pixelate`

### Tested environment
This project was tested in this environment with:
- Python: `3.14.0`
- OpenCV (Python `cv2`): `4.13.0`
- NumPy: `2.4.2`

### Files
- Python script: `yunet_webcam_face_blur.py`
- YuNet model (ONNX): `face_detection_yunet_2023mar.onnx`

### Download the YuNet ONNX model
Download from OpenCV Zoo:
- Source URL: [https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)

Save it into:
- `FaceBlurPixelate`

Example download command:
```bash
cd FaceBlurPixelate
curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o face_detection_yunet_2023mar.onnx
```

### Run the Python app
```bash
cd FaceBlurPixelate
source ../.venv/bin/activate
python yunet_webcam_face_blur.py --model face_detection_yunet_2023mar.onnx --mode blur
```

Pixelation example:
```bash
python yunet_webcam_face_blur.py --model face_detection_yunet_2023mar.onnx --mode pixelate --pixel-block-size 20
```

Optional arguments:
- `--camera 0`
- `--score-threshold 0.9`
- `--nms-threshold 0.3`
- `--top-k 5000`
- `--show-score`

Exit keys:
- `q`
- `Esc`

Note (macOS): if webcam access is denied, enable camera permission for your terminal/app in system privacy settings.

## 2. C++: How to Compile and Use

### What this does
This C++ app provides the same functionality as the Python app:
- YuNet face detection from webcam
- Face anonymization with `blur` or `pixelate`

### Files
- C++ source: `yunet_webcam_face_blur.cpp`
- Executable output: `yunet_webcam_face_blur`
- YuNet model (ONNX): `face_detection_yunet_2023mar.onnx`

### Download the YuNet ONNX model
Download from OpenCV Zoo:
- Source URL: [https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)

Example download command:
```bash
cd FaceBlurPixelate
curl -L "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" -o face_detection_yunet_2023mar.onnx
```

### Compile (OpenCV 4 with pkg-config)
```bash
cd FaceBlurPixelate
g++ -std=c++17 yunet_webcam_face_blur.cpp -o yunet_webcam_face_blur $(pkg-config --cflags --libs opencv4)
```

### Run the C++ app
```bash
cd FaceBlurPixelate
./yunet_webcam_face_blur --model face_detection_yunet_2023mar.onnx --mode blur
```

Pixelation example:
```bash
./yunet_webcam_face_blur --model face_detection_yunet_2023mar.onnx --mode pixelate --pixel-block-size 20
```

Optional arguments:
- `--camera 0`
- `--score-threshold 0.9`
- `--nms-threshold 0.3`
- `--top-k 5000`
- `--show-score`

Exit keys:
- `q`
- `Esc`

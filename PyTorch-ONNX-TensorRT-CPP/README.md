# How To Run Inference Using TensorRT C++ API

**This repository contains code for [How To Run Inference Using TensorRT C++ API](https://learnopencv.com/how-to-run-inference-using-tensorrt-c-api/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/25lt4zpeww9mmbg/AAC4iNJsUSNoUa1JUprhIQTQa?dl=1)

### To run PyTorch part:
```shell script
python3 -m pip install -r requirements.txt
python3 pytorch_model.py
```

### To run TensorRT part:
1. Install [CMake](https://cmake.org/) at least 3.10 version
2. Download and install NVIDIA CUDA 10.0 or later following by official instruction: [link](https://developer.nvidia.com/cuda-10.0-download-archive)
3. Download and extract CuDNN library for your CUDA version (login required): [link](https://developer.nvidia.com/rdp/cudnn-download)
4. Download and extract NVIDIA TensorRT library for your CUDA version (login required): [link](https://developer.nvidia.com/nvidia-tensorrt-6x-download). The minimum required version is 6.0.1.5
5. Add the path to CUDA, TensorRT, CuDNN to PATH variable (or LD_LIBRARY_PATH)
6. Build or install a pre-built version of OpenCV and OpenCV Contrib. The minimum required version is 4.0.0.

```shell script
mkdir build
cd build
cmake -DOpenCV_DIR=[path-to-opencv-build] -DTensorRT_DIR=[path-to-tensorrt] ..
make -j8
trt_sample[.exe] resnet50.onnx turkish_coffee.jpg
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
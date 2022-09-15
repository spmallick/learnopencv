
# Depth Estimation using Stereo Camera and OpenCV

**This repository contains code for [Depth Estimation using Stereo Camera and OpenCV](https://learnopencv.com/depth-perception-using-stereo-camera-python-c/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/cl70ubt31ya0e64/AABA77qwIX8rFiclw8drdPSXa?dl=1)

Create a custom low-cost stereo camera and capture depth maps with it using OpenCV.

## Directory Structure
**All the code files and folders follow the following structure.**

```
├── cpp
│   ├── disparity2depth_calib.cpp
│   ├── disparity_params_gui.cpp
│   ├── obstacle_avoidance.cpp
│   └── CMakeLists.txt
├── data
│   ├── depth_estimation_params.xml
│   ├── depth_estimation_params_cpp.xml
│   ├── depth_estmation_params_py.xml
│   ├── depth_params.xml
│   └── stereo_rectify_maps.xml
├── python
│   ├── disparity2depth_calib.py
│   ├── disparity_params_gui.py
│   ├── obstacle_avoidance.py
│   └── requirements.txt
└── README.md
```

## Instructions

### C++

To run the code in C++, please go into the `cpp` folder, then compile the `disparity_params_gui.cpp`, `obstacle_avoidance.cpp` and `disparity2depth_calib.cpp` code files, use the following:

```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Use the following commands to execute the compiled files:


```shell
./build/disparity_params_gui
./build/disparity2depth_calib
./build/obstacle_avoidance
```


### Python

To run the code in Python, please go into the `python` folder and refer to the following to use the `disparity_params_gui.py`, `obstacle_avoidance.py` and `disparity2depth_calib.py` files respectively:

```shell
python3 disparity_params_gui.py
python3 disparity2depth_calib.py
python3 obstacle_avoidance.py
```


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

# Contour Detection using OpenCV

**This repository contains code for [Contour Detection using OpenCV](https://learnopencv.com/contour-detection-using-opencv-python-c/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/lylz9lzqzs2nt2q/AAC5SfEV7_ex0imh1uip9-U6a?dl=1)

## Directory Structure

**All the code files and folders follow the following structure.**

```
├── CPP
│   ├── channel_experiments
│   │   ├── channel_experiments.cpp
│   │   └── CMakeLists.txt
│   ├── contour_approximations
│   │   ├── CMakeLists.txt
│   │   └── contour_approx.cpp
│   └── contour_extraction
│       ├── CMakeLists.txt
│       └── contour_extraction.cpp
├── input
│   ├── custom_colors.jpg
│   ├── image_1.jpg
│   └── image_2.jpg
├── python
│   ├── channel_experiments
│   │   └── channel_experiments.py
│   ├── contour_approximations
│   │   └── contour_approx.py
│   └── contour_extraction
│       └── contour_extraction.py
└── README.md
```



## Instructions

### Python

To run the code in Python, please go into the `python` folder and execute the Python scripts in each of the respective sub-folders.

### C++

To run the code in C++, please go into the `cpp` folder, then go into each of the respective sub-folders and follow the steps below:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/channel_experiments
```

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/contour_approximations
```

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/contour_extraction
```



# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/18c5719ef10afe9607af3e87e990068c942ae4cba8bd4d72d21950d6213ea97e/68747470733a2f2f7777772e6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30342f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)

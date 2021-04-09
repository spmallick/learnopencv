# Deep Learning with OpenCV's DNN Module



## Directory Structure

**All the code files and folders follow the following structure.**

```
├── cpp
│   ├── classify
│   │   ├── classify.cpp
│   │   └── CMakeLists.txt
│   └── detection
│       ├── detect_img
│       │   ├── CMakeLists.txt
│       │   └── detect_img.cpp
│       └── detect_vid
│           ├── CMakeLists.txt
│           └── detect_vid.cpp
├── input
│   ├── classification_classes_ILSVRC2012.txt
│   ├── DenseNet_121.caffemodel
│   ├── DenseNet_121.prototxt
│   ├── frozen_inference_graph.pb
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── object_detection_classes_coco.txt
│   ├── ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt
│   └── video_1.mp4
├── outputs
│   ├── image_result.jpg
│   ├── result_image.jpg
│   └── video_result.mp4
├── python
│   ├── classification
│   │   ├── classify.py
│   │   └── README.md
│   ├── detection
│   │   ├── detect_img.py
│   │   └── detect_vid.py
│   └── requirements.txt
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
./build/classify
```

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/detect_img
```

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/detect_vid
```



## Outputs

### Image Classification

* ![](outputs/result_image.jpg)

### Object Detection

* ![](outputs/image_result.jpg)

  

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/18c5719ef10afe9607af3e87e990068c942ae4cba8bd4d72d21950d6213ea97e/68747470733a2f2f7777772e6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30342f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)
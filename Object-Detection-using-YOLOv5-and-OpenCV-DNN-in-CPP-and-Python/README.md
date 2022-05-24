# Object Detection using YOLOv5 and OpenCV DNN (C++/Python)

The repository contains code for the tutorial [Object Detection using yolov5 in OpenCV DNN framework](https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/). Use this 
[Colab notebook](https://colab.research.google.com/github/spmallick/learnopencv/blob/master/Object-Detection-using-YOLOv5-and-OpenCV-DNN-in-CPP-and-Python/Convert_PyTorch_models.ipynb) to convert native PyTorch models into ONNX format.

![Feature Image](https://learnopencv.com/wp-content/uploads/2022/04/yolov5-feature-image.gif)

## Install Dependancies

```
pip install -r requirements.txt
```
List of tutorials for installing OpenCV for C++ [here](https://learnopencv.com/category/install/).


## Execution
### Python
```Python
python yolov5.py
```
### CMake C++ Linux
```C++ Linux
mkdir build
cd build
cmake ..
cmake --build .
./main
```
### CMake C++ Windows
```C++ Windows
rmdir /s /q build
cmake -S . -B build
cmake --build build --config Release
.\build\Release\main.exe
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/" target="_blank">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

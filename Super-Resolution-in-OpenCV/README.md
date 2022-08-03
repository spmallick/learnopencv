# Super Resolution in OpenCV
**This repository contains code for [Super Resolution in OpenCV](https://learnopencv.com/super-resolution-in-opencv/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/na6ygbayem1iico/AAB3XpS9wfaJDMJIJMToqaKHa?dl=1)

Please note that the code requires installation of OpenCV Contrib module along with the core modules.

The model files used in the example code can be downloaded from the links below.

* [EDSR](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models)
* [ESPCN](https://github.com/fannymonori/TF-ESPCN/tree/master/export)
* [FSRCNN](https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models)
* [LapSRN](https://github.com/fannymonori/TF-LapSRN/tree/master/export)

You should also check out [opencv_contrib repository](https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres) for more details on OpenCV Super Resolution


## Instructions

### Python

To run the code in Python, please use `super_res.py`

### C++

To run the code in C++, please follow the steps given below:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
./build/SuperResolution
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

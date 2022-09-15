
# OpenCV QR Code Scanner ( C++ and Python )

**This repository contains the code for [OpenCV QR Code Scanner ( C++ and Python )](https://learnopencv.com/opencv-qr-code-scanner-c-and-python/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/195xuwdmt1nbpor/AAAucMK_7Khcch9U9HrLX3iqa?dl=1)

This code requires **OpenCV 3.4.4 or above** or later. 

# For C++

## How to compile the code

Specify the **OpenCV_DIR** in CMake option

```
cmake -D OpenCV_DIR=<path to opencv install directory>/lib/cmake/opencv4/ .
make
```

OR First Specify the **OpenCV_DIR** in CMakeLists.txt file. Then,

```
cmake .
make
```
# How to Run the code

## C++ ##
```
./qrCodeOpencv <filename>
```
## Python ##
```
python qrCodeOpencv.py <filename>
```
**Note** : If you dont give any filename, it will use the default image provided.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

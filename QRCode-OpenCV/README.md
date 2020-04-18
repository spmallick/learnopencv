
# OpenCV QR Code Scanner

This directory contains code for using OpenCV QR code. This code requires **OpenCV 3.4.4 or above** or later. 

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

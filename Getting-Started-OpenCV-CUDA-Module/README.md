
# Getting Started with OpenCV CUDA Module

**This repository contains code for [Getting Started with OpenCV CUDA Module](https://www.learnopencv.com/getting-started-opencv-cuda-module/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/1ijilsp7m6rmx7q/AABUVfmaK2DYL2QI-89lqoDma?dl=1)

## Getting Started

Our code is tested using Python 3.7.5, but it should also work with any other python3.x. If you'd like to check your
version run:

```bash
python3 -V
```

### Virtual Environment

Let's create a new virtual environment. You'll need to install [virtualenv](https://pypi.org/project/virtualenv/)
package if you don't have it:

```bash
pip install virtualenv
```

Now we can create a new virtualenv variable and call it `env`:

```bash
python3 -m venv env
```

The last thing we have to do is to activate it:

```bash
source  env/bin/activate
```

### Numpy

Install numpy package by running:

```bash
pip install numpy
```

### Installing CUDA

The code was tested using CUDA Toolkit 10.2. Please follow the official instruction to download [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive) or higher.

### OpenCV with CUDA Support

In this blog post, we're using OpenCV with CUDA support to accelerate OpenCV algorithms. That is why we will need to customize the OpenCV library build and make it from scratch. To do so:

1. Install dependencies:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake unzip pkg-config
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libgtk-3-dev
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install python3-dev

```

2. Download the latest OpenCV version from the official repository:

```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
```

3. Unzip the downloaded archives:

```bash
unzip opencv.zip
unzip opencv_contrib.zip
```

4. Rename the directories to match CMake paths:

```bash
mv opencv-4.4.0 opencv
mv opencv_contrib-4.4.0 opencv_contrib
```

5. Compile OpenCV:

Create and enter a build directory:

```bash
cd opencv
mkdir build
cd build
```

Run CMake to configure the OpenCV build. Don't forget to set the right pass to the `PYTHON_EXECUTABLE`. If you are using the CUDA version different from `10.2`, please change the last 3 arguments accordingly.

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D PYTHON_EXECUTABLE=env/bin/python3 \
      -D BUILD_EXAMPLES=ON \
      -D WITH_CUDA=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 \
      -D OpenCL_LIBRARY=/usr/local/cuda-10.2/lib64/libOpenCL.so \
      -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.2/include/ \
      ..
```

Check the output and make sure that everything is set correctly. After that we're ready to build it with:

```bash
make -j4
```

Make sure, you didn't get any errors. Then run the following command:

```bash
sudo ldconfig
```

which creates the necessary links and cache to our freshly built shared library.
Rename the created Python3 bindings for OpenCV to `cv2.so`:

```bash
mv lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so cv2.so
```

The last step is to create a symlink of our OpenCV `cv2.so` into the virtual environment installed packages:

```bash
cd env/lib/python3.7/site-packages/
ln -s ~/opencv/build/cv2.so cv2.so
```

## Running the Demo

**C++**

You first need to compile .cpp file with the following command:

```bash
g++ `pkg-config --cflags --libs opencv4` demo.cpp -o demo.out -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_cudaarithm -lopencv_cudaoptflow -lopencv_cudaimgproc -lopencv_cudawarping -std=c++11

```

After that to run the demo, you will need to pass:

- path to the video file,
- device, to choose between CPU and GPU inference. By default, the device is set to "cpu".

For example:

```bash
./demo.out video/boat.mp4 gpu
```

**Python**

To run the demo, you will need to pass:

- `--video` argument to set the path to the video file,
- `--device` to choose between CPU and GPU inference. By default, the device is set to "cpu".

For example:

```bash
python3 demo.py --video video/boat.mp4 --device "cpu"
```


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

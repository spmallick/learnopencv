# Background Subtraction with OpenCV and BGS Libraries

**This repository contains code for [Background Subtraction with OpenCV and BGS Libraries](https://learnopencv.com/background-subtraction-with-opencv-and-bgs-libraries) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/x2qyr29bcj1euu8/AAAyiHdaRlMKo5VOujrHacY8a?dl=1)

## Getting Started

Please, follow the instructions to set up the environment for Linux-based systems. This installation guide is based on [the official instruction](https://github.com/andrewssobral/bgslibrary/wiki/Wrapper:-Python#building-bgslibrary-with-python-support-on-linux), with some additions and corrections.

If you are using Windows OS follow the [Building BGSLibrary with Python support on Windows](https://github.com/andrewssobral/bgslibrary/wiki/Wrapper:-Python#building-bgslibrary-with-python-support-on-windows) section from the same instruction.

Our code is tested using Python 3.7.5, but it should also work with any other python3.x. If you'd like to check your version run:

```bash
python3 -V
```

_Note:_ We assume, that your current location is `learnopencv/Background-Subtraction` and will refer it as `work_dir`.

### Virtual Environment

Let's create a new virtual environment. You'll need to install [virtualenv](https://pypi.org/project/virtualenv/) package if you don't have it:

```bash
pip install virtualenv
```

Now we can create a new virtualenv variable and call it `env`:

```bash
python3.7 -m virtualenv venv
```

The last thing we have to do is to activate it:

```bash
source  venv/bin/activate
```
To will need also to install numpy package:

```bash
pip install numpy
```

### OpenCV

In this blog post we are using BGS Library, which is heavily based on OpenCV. That is why, we first need to build the OpenCV library. To do so:
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
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.3.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.3.0.zip
```

3. Unzip the downloaded archives:

```bash
unzip opencv.zip
unzip opencv_contrib.zip
```

4. Rename the directories to match CMake paths:

```bash
mv opencv-4.3.0 opencv
mv opencv_contrib-4.3.0 opencv_contrib
```

5. Compile OpenCV Create and enter a build directory:

```bash
cd ~/opencv
mkdir build
cd build
```

Run CMake to configure the OpenCV build. Don't forget to set the right pass to the `PYTHON_EXECUTABLE`.

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D PYTHON_EXECUTABLE=work_dir/venv/bin/python3 \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=ON ..
```

Check the output and make sure that everything is set correctly. After that we are ready to build it with:

```bash
make -j4
```

Make sure, you didn't get any errors. Then run the following commands:

```bash
sudo make install
sudo ldconfig
```

which creates the necessary links and cache to our freshly-built shared library.

Put `lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so` into the virtual environment installed packages:

```bash
cp lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so work_dir/venv/lib/python3.7/site-packages/cv2.so
```

### BGS Library

Now everything is ready to build BGS Library.

1. Download the source code:

```bash
cd work_dir
git clone --recursive https://github.com/andrewssobral/bgslibrary.git
```

2. Make `build` folder and navigate to it:

```bash
cd bgslibrary
mkdir build && cd build
```

3. Run CMake to configure the build. Don't forget to set `PYTHON_EXECUTABLE` to your virtual environment python.

```bash
cmake -D BGS_PYTHON_SUPPORT=ON\
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OpenCV_DIR=~/opencv/build \
  -D PYTHON_EXECUTABLE=work_dir/venv/bin/python  ..
```

4. Check the output. Pay attention to the `Python library status` section. It should look similar to this:

```bash
-- Python library status:
--     executable: ~/env/bin/python
--     library: ~/.pyenv/versions/3.7.5/lib/libpython3.7m.so
--     include path: ~/.pyenv/versions/3.7.5/include/python3.7m
```

Make sure, that your python library is build as a shared library (.so file), not as a static (.a file). That might cause
an error if you are using [pyenv](https://github.com/pyenv/pyenv), that builds python library as a static library by
default. To rebuild it as a shared library, run:

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install --force 3.7.5
```

5. Build the BGS Library:

```bash
make -j4
```

6. Make sure, you didn't get any errors. You can check, that everything is working by running the demo script:

```bash
cd ..
python3 demo.py
```

## Running the Demo

### Python

The proposed for experiments `background_subtr_opencv.py` and `background_subtr_bgslib.py` scripts support `--input_video` key to customize the background subtraction pipeline. `--input_video` contains the path to the input video. By default its value is `"space_traffic.mp4"`. Please, follow the below instructions for each case.

#### OpenCV Library

To run OpenCV case, please, choose one of the described below scenarios:

- for the default input video:

```bash
python3 background_subtr_opencv.py
```

- for the custom input video:

```bash
python3 background_subtr_opencv.py --input_video <VIDEO_PATH>
```

#### BGS Library

To run BGSLibrary case, please, choose one of the described below scenarios:

- for the default input video:

```bash
python3 background_subtr_bgslib.py
```

- for the custom input video:

```bash
python3 background_subtr_bgslib.py --input_video <VIDEO_PATH>
```

### C++

The first step is compiling of the proposed `.cpp` files with the appropriate commands for each case.

#### OpenCV Library

To compile `background_subtr_opencv.cpp` you need to run the below command:

```bash
g++ background_subtr_opencv.cpp `pkg-config --cflags --libs opencv4` -o background_subtr_opencv.out -std=c++11
```

After `background_subtr_opencv.out` was obtained, we can run the BS-pipeline:

```bash
./background_subtr_opencv.out
```

By default `space_traffic.mp4` will be used. To provide another video as input, you need to define `--input` key value:

```bash
./background_subtr_opencv.out --input=<VIDEO_PATH>
```

#### BGS Library

To compile `background_subtr_bgslib.cpp` you need to run the below command:

```bash
g++ background_subtr_bgslib.cpp `pkg-config --cflags --libs opencv4` -lbgslibrary_core  -I bgslibrary/src -o background_subtr_bgslib.out -std=c++11
```

After `background_subtr_bgslib.out` was obtained, we can run the BS-pipeline:

```bash
./background_subtr_bgslib.out
```

By default `space_traffic.mp4` will be used. To provide another video as input, you need to define `--input` key value:

```bash
./background_subtr_bgslib.out --input=<VIDEO_PATH>
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

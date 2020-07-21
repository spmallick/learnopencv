## Getting Started

Please, follow the instructions to set up the environment for Linux-based systems. This installation guide is based on
[the official instruction](https://github.com/andrewssobral/bgslibrary/wiki/Wrapper:-Python#building-bgslibrary-with-python-support-on-linux),
with some additions and corrections.

If you are using Windows OS follow the
[Building BGSLibrary with Python support on Windows](https://github.com/andrewssobral/bgslibrary/wiki/Wrapper:-Python#building-bgslibrary-with-python-support-on-windows)
section from the same instruction.

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
python3 -m venv ~/env
```

The last thing we have to do is to activate it:

```bash
source  ~/env/bin/activate
```

### Numpy

Install numpy package by running:

```bash
pip install numpy
```

### OpenCV

In this blog post we are using BGS Library, which is heavily based on OpenCV. That is why, we first need to build the
OpenCV library. To do so:

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
cmake -D CMAKE_BUILD_TYPE=RELEASE\
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D INSTALL_PYTHON_EXAMPLES=OFF \
 -D INSTALL_C_EXAMPLES=OFF \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
 -D PYTHON_EXECUTABLE=~/env/bin/python3 \
 -D BUILD_EXAMPLES=ON ..

```

Check the output and make sure that everything is set correctly. After that we are ready to build it with:

```bash
make -j4
```

Make sure, you didn't get any errors. Then run the following command:

```bash
sudo ldconfig
```

which creates the necessary links and cache to our freshly-built shared library.

Rename the created Python 3 bindings for OpenCV to `cv2.so`:

```bash
mv lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so cv2.so
```

The last step is to create a symlink of our OpenCV `cv2.so` into our cv virtual environment:

```bash
cd ~/env/lib/python3.7/site-packages/
ln -s ~/opencv/build/cv2.so cv2.so
```

### BGS Library

Now everything is ready to build BGS Library.

1. Download the source code:

```bash
cd build
git clone --recursive https://github.com/andrewssobral/bgslibrary.git
```

2. Run CMake to configure the build. Don't forget to set `PYTHON_EXECUTABLE` to your virtual environment python.

```bash
cmake -D BGS_PYTHON_SUPPORT=ON\
  -D OpenCV_DIR=~/opencv/build \
  -D PYTHON_EXECUTABLE=~/env/bin/python  ..
```

3. Check the output. Pay attention to the `Python library status` section. It should look similar to this:

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

4. Build the BGS Library:

```bash
make -j4
```

5. Check, that everything is working by running the demo script:

```bash
cd ..
python3 demo.py
```

## Executing Scripts
The proposed for experiments ``background_subtr_opencv.py`` and ``background_subtr_bgslib.py`` scripts support ``--input_video`` key to customize the background subtraction pipeline.
``--input_video`` contains the path to the input video. By default its value is ``"space_traffic.mp4"``. Please, follow the below instructions for each case.

### OpenCV Library
To run OpenCV case, please, choose one of the described below scenarios:
* for the default input video:

```bash
python3 background_subtr_opencv.py
```

* for the custom input video:

```bash
python3 background_subtr_opencv.py --input_video <VIDEO_PATH>
```

### BGS Library
To run BGSLibrary case, please, choose one of the described below scenarios:
* for the default input video:

```bash
python3 background_subtr_bgslib.py
```

* for the custom input video:

```bash
python3 background_subtr_bgslib.py --input_video <VIDEO_PATH>
```

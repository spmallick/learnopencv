This contains the code for **Image Classification with OpenCV for Android**. For more information - visit [**Image Classification with OpenCV for Android**](https://www.learnopencv.com/image-classification-with-opencv-for-android/)


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
python3 -m venv ~/env
```

The last thing we have to do is to activate it:

```bash
source  ~/env/bin/activate
```

To install the required python dependencies run:

```bash
pip3 install -r requirements.txt
```

### OpenCV

In this blog post we are using OpenCV 4.3.0 unavailable via `pip`. The first step is building the OpenCV library. To do so:

1. Check the list of the below libraries. Install the missed dependencies:

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

For OpenCV Java installation we used default Java Runtime Environment and Java Development Kit:

```bash
sudo apt-get install default-jre
sudo apt-get install default-jdk
sudo apt-get install ant
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

5. Compile OpenCV. Create and enter a build directory:

```bash
cd ~/opencv
mkdir build && cd build
```

6. Run CMake to configure the OpenCV build. Don't forget to set the right pass to the ``PYTHON_EXECUTABLE``:

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D INSTALL_PYTHON_EXAMPLES=OFF \
 -D INSTALL_C_EXAMPLES=OFF \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
 -D PYTHON_EXECUTABLE=~/env/bin/python3 \
 -D ANT_EXECUTABLE=/usr/bin/ant \
 -D BUILD_SHARED_LIBRARY=OFF \
 -D BUILD_TESTS=OFF \
 -D BUILD_PERF_TESTS=OFF \
 -D BUILD_EXAMPLES=ON ..
```

If you want to configure the build with some specific Java version, please, add the following fields, verifying the paths:

```bash
 -D JAVA_AWT_INCLUDE_PATH=/usr/lib/jvm/java-1.x.x-openjdk-amd64/include \
 -D JAVA_AWT_LIBRARY=/usr/lib/jvm/java-1.x.x-openjdk-amd64/lib/libawt.so \
 -D JAVA_INCLUDE_PATH=/usr/lib/jvm/java-1.x.x-openjdk-amd64/include \
 -D JAVA_INCLUDE_PATH2=/usr/lib/jvm/java-1.x.x-openjdk-amd64/include/linux \
 -D JAVA_JVM_LIBRARY=/usr/lib/jvm/java-1.x.x-openjdk-amd64/include/jni.h \
```

7. Check the output and make sure that everything is set correctly. After that we're ready to build it with:

```bash
make -j8
```

Make sure, you didn't get any errors. In case of successful completion you will find the following files in the ``build`` directory ``lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so``.

Then run the following command:

```bash
sudo ldconfig
```

which creates the necessary links and cache to our freshly built shared library.

The last step is to move ``lib/python3/cv2.cpython-37m-x86_64-linux-gnu.so`` into the virtual environment installed packages:

```bash
cp lib/python3/cv2.cpython-36m-x86_64-linux-gnu.so ~/env/lib/python3.7/site-packages/cv2.so
```

### OpenCV Android

For Android application development we will need [OpenCV for Android](https://github.com/opencv/opencv/releases/tag/4.3.0):

```bash
wget https://github.com/opencv/opencv/releases/download/4.3.0/opencv-4.3.0-android-sdk.zip -O opencv-4.3.0-android-sdk.zip
unzip opencv-4.3.0-android-sdk.zip
rm opencv-4.3.0-android-sdk.zip
```

## Executing Model Conversion and Test Script
The proposed for the experiments ``MobileNetV2ToOnnx.py`` script supports the ``--input_image`` key to customize the model conversion pipeline. It defines the full input image path, including its name - ``"test_img_cup.jpg"`` by default.

To run MobileNetV2 conversion case, please, choose one of the described below scenarios:

* for the custom input image and running evaluation of the converted model:

```bash
python3 MobileNetV2Conversion.py --input_image <image_name>
```

* for the default input image:

```bash
python3 MobileNetV2Conversion.py
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

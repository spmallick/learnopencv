#!/bin/bash

############## WELCOME #############
# Step 0: Take inputs

# Clean build directories
rm -rf opencv/build
rm -rf opencv_contrib/build

echo "OpenCV installation by learnOpenCV.com"

echo "Select OpenCV version to install (1 or 2)"
echo "1. OpenCV 3.4.3 (default)"
echo "2. Master"

read cvVersionChoice

if [ "$cvVersionChoice" -eq 2 ]; then
        cvVersion="master"
else
	cvVersion="3.4.3"
fi

# Create directory for installation
mkdir installation
mkdir installation/OpenCV-"$cvVersion"

# Save current working directory
cwd=$(pwd)

# Step 1: Update packages
echo "Updating packages"

sudo apt -y update
sudo apt -y upgrade
echo "================================"

echo "Complete"

# Step 2: Install OS libraries
echo "Installing OS libraries"

sudo apt -y remove x264 libx264-dev

## Install dependencies
sudo apt -y install build-essential checkinstall cmake pkg-config yasm
sudo apt -y install git gfortran
sudo apt -y install libjpeg8-dev libjasper-dev libpng12-dev

sudo apt -y install libtiff5-dev

sudo apt -y install libtiff-dev

sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt -y install libxine2-dev libv4l-dev
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd $cwd

sudo apt -y install libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt -y install libgtk2.0-dev libtbb-dev qt5-default
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libvorbis-dev libxvidcore-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libavresample-dev
sudo apt -y install x264 v4l-utils

# Optional dependencies
sudo apt -y install libprotobuf-dev protobuf-compiler
sudo apt -y install libgoogle-glog-dev libgflags-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
echo "================================"

echo "Complete"


# Step 3: Install Python libraries
echo "Install Python libraries"

sudo apt -y install python-dev python-pip python3-dev python3-pip
sudo -H pip2 install -U pip numpy
sudo -H pip3 install -U pip numpy
sudo apt -y install python3-testresources

# Install virtual environment
sudo -H pip2 install virtualenv virtualenvwrapper
sudo -H pip3 install virtualenv virtualenvwrapper
echo "# Virtual Environment Wrapper" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
#source ~/.bashrc
cd $cwd
source /usr/local/bin/virtualenvwrapper.sh
echo "================================"

echo "Complete"

echo "Creating Python environments"

############ For Python 2 ############
# create virtual environment
mkvirtualenv OpenCV-"$cvVersion"-py2 -p python2
workon OpenCV-"$cvVersion"-py2

# now install python libraries within this virtual environment
pip install numpy scipy matplotlib scikit-image scikit-learn ipython

# quit virtual environment
deactivate
######################################

############ For Python 3 ############
# create virtual environment
mkvirtualenv OpenCV-"$cvVersion"-py3 -p python3
workon OpenCV-"$cvVersion"-py3

# now install python libraries within this virtual environment
pip install numpy scipy matplotlib scikit-image scikit-learn ipython

# quit virtual environment
deactivate
######################################
echo "================================"
echo "Complete"

# Step 4: Download opencv and opencv_contrib
echo "Downloading opencv and opencv_contrib"
git clone https://github.com/opencv/opencv.git
cd opencv
if [ "$cvVersionChoice" -eq 2 ]; then
	git checkout $cvVersion
else
	git checkout tags/3.4.3
fi
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
if [ "$cvVersionChoice" -eq 2 ]; then
	git checkout $cvVersion
else
	git checkout tags/3.4.3
fi
cd ..
echo "================================"
echo "Complete"

# Step 5: Compile and install OpenCV with contrib modules
echo "================================"
echo "Compiling and installing OpenCV with contrib modules"
cd opencv
mkdir build
cd build

# For system wide installation:
# Change CMAKE_INSTALL_PREFIX=/usr/local \

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	        -D CMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV-"$cvVersion" \
	        -D INSTALL_C_EXAMPLES=ON \
	        -D INSTALL_PYTHON_EXAMPLES=ON \
	        -D WITH_TBB=ON \
	        -D WITH_V4L=ON \
		-D WITH_QT=ON \
		-D WITH_OPENGL=ON \
		-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
		-D BUILD_EXAMPLES=ON ..

make -j4
make install

# Create symlink in virtual environment
py2binPath=$(find $cwd/installation/OpenCV-$cvVersion/lib/ -type f -name "cv2.so")
py3binPath=$(find $cwd/installation/OpenCV-$cvVersion/lib/ -type f -name "cv2.cpython*.so")

# Link the binary python file
cd ~/.virtualenvs/OpenCV-$cvVersion-py2/lib/python2.7/site-packages/
ln -f -s $py2binPath cv2.so

cd ~/.virtualenvs/OpenCV-$cvVersion-py3/lib/python3.5/site-packages/
ln -f -s $py3binPath cv2.so

# Step 6: Compile and install dlib
echo "================================"
echo "Compiling and installing dlib"
cd $cwd
wget http://dlib.net/files/dlib-19.6.tar.bz2
tar xvf dlib-19.6.tar.bz2
cd dlib-19.6/
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd ..

pkg-config --libs --cflags dlib-1


############ For Python 2 ############
workon OpenCV-"$cvVersion"-py2
cd $cwd
cd dlib-19.6
python setup.py install
rm -rf dist
rm -rf tools/python/build
rm python_examples/dlib.so

cd ..

############ For Python 3 ############
workon OpenCV-"$cvVersion"-py3
cd $cwd
cd dlib-19.6
python setup.py install
rm -rf dist
rm -rf tools/python/build
rm python_examples/dlib.so

cd $cwd

# Print instructions
echo "================================"
echo "Installation complete. Printing test instructions."

echo workon OpenCV-"$cvVersion"-py2
echo "ipython"
echo "import cv2"
echo "cv2.__version__"

if [ $cvVersionChoice -eq 2 ]; then
	       echo "The output should be 4.0.0-pre"
else
               echo The output should be "$cvVersion"
fi

echo "deactivate"

echo workon OpenCV-"$cvVersion"-py3
echo "ipython"
echo "import cv2"
echo "cv2.__version__"

if [ $cvVersionChoice -eq 2 ]; then
              echo "The output should be 4.0.0-pre"
else
              echo The output should be "$cvVersion"
fi

echo "deactivate"

echo "Installation completed successfully"

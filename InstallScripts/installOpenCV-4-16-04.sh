#!/bin/bash

echo "OpenCV installation by learnOpenCV.com"
 
# Define OpenCV version to install
cvVersion="master"

# Clean build directories
rm -rf opencv/build
rm -rf opencv_contrib/build

# Create directory for installation
mkdir installation
mkdir installation/OpenCV-"$cvVersion"

# Save current working directory
cwd=$(pwd)

sudo apt -y update
sudo apt -y upgrade

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

sudo apt -y install python3-dev python3-pip
sudo -H pip3 install -U pip numpy
sudo apt -y install python3-testresources

# Install virtual environment
sudo -H pip3 install virtualenv virtualenvwrapper
echo "# Virtual Environment Wrapper" >> ~/.bashrc
VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
echo "VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
cd $cwd
source /usr/local/bin/virtualenvwrapper.sh

############ For Python 3 ############
# create virtual environment
mkvirtualenv OpenCV-"$cvVersion"-py3 -p python3
workon OpenCV-"$cvVersion"-py3
 
# now install python libraries within this virtual environment
pip install numpy scipy matplotlib scikit-image scikit-learn ipython dlib
 
# quit virtual environment
deactivate
######################################

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout $cvVersion
cd ..
 
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout $cvVersion
cd ..


cd opencv
mkdir build
cd build

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
py3binPath=$(find $cwd/installation/OpenCV-$cvVersion/python/cv2/python-3.5/ -type f -name "cv2.cpython*.so")
 
# Link the binary python file
cd ~/.virtualenvs/OpenCV-$cvVersion-py3/lib/python3.5/site-packages/
ln -f -s $py3binPath cv2.so

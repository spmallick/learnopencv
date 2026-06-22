#!/bin/bash

############## WELCOME #############
# Get command line argument for verbose
echo "Welcome to OpenCV Installation Script for Raspbian Stretch"
echo "This script is provided by LearnOpenCV.com"

######### VERBOSE ON ##########

# Step 0: Take inputs
echo "OpenCV installation by learnOpenCV.com"

cvVersion="master"

# Clean build directories
rm -rf opencv/build
rm -rf opencv_contrib/build

# Create directory for installation
mkdir installation
mkdir installation/OpenCV

# Save current working directory
cwd=$(pwd)

# Step 1: Update packages
echo "Updating packages"

sudo apt-get -y update
sudo apt-get -y upgrade
echo "================================"

echo "Complete"

# Step 2: Install OS libraries
echo "Installing OS libraries"

# compiler and build tools
sudo apt-get -y install git build-essential cmake pkg-config checkinstall
# development files for system wide image codecs
sudo apt-get -y install libjpeg-dev libpng-dev libtiff-dev
# Protobuf library and tools for dnn module
sudo apt-get -y install libprotobuf-dev protobuf-compiler
# development files for V4L2 to enable web cameras support in videoio module
sudo apt-get -y install libv4l-dev

# Optional dependencies

# FFmpeg development files to enable video decoding and encoding in videoio module
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev
# development files for GTK 2.0 UI library for highgui module
sudo apt-get -y install libgtk2.0-dev
# Eigen library needed for some modules in contrib repository
sudo apt-get -y install libeigen3-dev
# Numpy and Python3 development files for Python bindings
sudo apt-get -y install python3-dev python3-pip
echo "================================"

echo "Complete"


# Step 3: Install Python libraries
echo "Install Python libraries"

sudo -H pip3 install numpy

echo "================================"

echo "Complete"

sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/g' /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

echo "================================"
echo "Complete"

# Step 4: Download opencv and opencv_contrib
echo "Downloading opencv and opencv_contrib"
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout master
cd ..

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout master
cd ..

echo "================================"
echo "Complete"

# Step 5: Compile and install OpenCV with contrib modules
echo "================================"
echo "Compiling and installing OpenCV with contrib modules"
cd opencv
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$cwd/installation/OpenCV \
    -DBUILD_EXAMPLES=ON
    -DINSTALL_C_EXAMPLES=ON \
    -DINSTALL_PYTHON_EXAMPLES=ON \
    -DWITH_TBB=ON \
    -DENABLE_NEON=ON \
    -DENABLE_VFPV3=ON \
    -DWITH_V4L=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules

make
make install

cd $cwd

sudo sed -i 's/CONF_SWAPSIZE=1024/CONF_SWAPSIZE=100/g' /etc/dphys-swapfile
sudo /etc/init.d/dphys-swapfile stop
sudo /etc/init.d/dphys-swapfile start

#!/bin/sh

echo "OpenCV installation by learnOpenCV.com"

echo "Make sure XCode is installed"

echo "Press Enter if you have installed XCode"

read tmp

echo "Installing Homebrew"

ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# update homebrew 
brew update
  
# Add Homebrew path in PATH
echo "# Homebrew" >> ~/.bash_profile
echo "export PATH=/usr/local/bin:$PATH" >> ~/.bash_profile
#source ~/.bash_profile

brew install python3
brew install cmake
brew install qt5

QT5PATH=/usr/local/Cellar/qt/5.11.2_1
# Save current working directory
cwd=$(pwd)

sudo -H pip3 install -U pip numpy

# Install virtual environment
sudo -H python3 -m pip install virtualenv virtualenvwrapper
VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3
echo "VIRTUALENVWRAPPER_PYTHON=/usr/local/bin/python3" >> ~/.bash_profile
echo "# Virtual Environment Wrapper" >> ~/.bash_profile
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bash_profile
cd $cwd
source /usr/local/bin/virtualenvwrapper.sh

#=================================================================

echo "Installing OpenCV - 3.4.x"
 
#Specify OpenCV version
cvVersion="3.4"

# Clean build directories
rm -rf opencv/build
rm -rf opencv_contrib/build

# Create directory for installation
mkdir installation
mkdir installation/OpenCV-"$cvVersion"

############ For Python 3 ############
# create virtual environment
mkvirtualenv OpenCV-"$cvVersion"-py3 -p python3
workon OpenCV-"$cvVersion"-py3
 
# now install python libraries within this virtual environment
pip install cmake numpy scipy matplotlib scikit-image scikit-learn ipython dlib
 
# quit virtual environment
deactivate
######################################

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout "$cvVersion"
cd ..
 
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout "$cvVersion"
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
            -D OPENCV_SKIP_PYTHON_LOADER=ON \
            -D CMAKE_PREFIX_PATH=$QT5PATH \
            -D CMAKE_MODULE_PATH="$QT5PATH"/lib/cmake \
            -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/OpenCV-"$cvVersion"-py3/lib/python3.7/site-packages \
        -D WITH_QT=ON \
        -D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON ..
        
make -j$(sysctl -n hw.physicalcpu)
make install

cd $cwd

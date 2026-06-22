#!/bin/sh

set -e -x

if [ -f "CMakeLists.txt" ]; then
    echo "Using existing CMakeLists.txt file"
    clean_cmake=0
else
    echo "Copy default CMakeLists.txt file"
    cp ../CI/CMakeLists.txt .
    clean_cmake=1
fi
rm -rf build
mkdir -p build
cd build
cmake -DOpenCV_DIR=$OPENCV_DIR ../
make
cd ..
rm -rf build
if [ $clean_cmake -ne 0 ]; then
    echo "Removing copied cmake"
    rm -rf CMakeLists.txt
fi

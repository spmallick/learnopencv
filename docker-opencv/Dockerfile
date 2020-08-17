FROM python:3.7.8-slim

MAINTAINER Labhesh Valechha <labheshvalechha@gmail.com>

ARG opencv_version=4.4.0
ARG pythonVersion=3.7
ENV DEBIAN_FRONTEND noninteractive

ENV WORK_DIR "/home"
WORKDIR $WORK_DIR

RUN apt-get update && apt-get upgrade -y \
        && apt-get install -y --no-install-recommends apt-utils \
        && apt-get install -y wget

RUN python -m pip install --upgrade pip

RUN apt-get install -y cmake git vim libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libeigen3-dev libgflags-dev libgoogle-glog-dev libhdf5-dev \
        && pip install numpy

ENV PYTHONPATH=/usr/local/lib/python"$pythonVersion"/
ENV PYTHONHOME=/usr/local

RUN git clone https://github.com/opencv/opencv_contrib.git \
        && cd opencv_contrib \
        && git checkout $opencv_version \
        && cd ..\
        && git clone https://github.com/opencv/opencv.git \
        && cd opencv \
        && git checkout $opencv_version\
        && mkdir build \
        && cd build \
        && cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D BUILD_EXAMPLES=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python"$pythonVersion"/site-packages/numpy/core/include/ \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        .. \
        && make install/strip \
        && cd $WORK_DIR \
        && rm -rf opencv opencv_contrib

WORKDIR $WORK_DIR

RUN mkdir Sample-Code-Cpp-Unix && cd Sample-Code-Cpp-Unix \
        && mkdir build \
        && wget https://raw.githubusercontent.com/spmallick/learnopencv/master/docker-opencv/Sample-Code-Cpp-Unix/boy.jpg \
        && wget https://raw.githubusercontent.com/spmallick/learnopencv/master/docker-opencv/Sample-Code-Cpp-Unix/CMakeLists.txt \
        && wget https://raw.githubusercontent.com/spmallick/learnopencv/master/docker-opencv/Sample-Code-Cpp-Unix/sampleCode.cpp

RUN apt-get -qq -y autoremove \
        && apt-get -qq autoclean

CMD ["bash"]
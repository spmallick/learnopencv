FROM python:3.7-slim-buster

MAINTAINER Labhesh Valechha <labheshvalechha@gmail.com>

ARG opencv_version=4.4.0
ARG pythonVersion=3.7
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	apt-utils \
	build-essential \
	cmake \
	git \
	vim \
	libgtk2.0-dev \
	pkg-config \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libeigen3-dev \
	libgflags-dev \
	libgoogle-glog-dev \
	libhdf5-dev \
	&& pip install numpy \
	&& rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/usr/local/lib/python"$pythonVersion"/
ENV PYTHONHOME=/usr/local

WORKDIR /home

RUN git clone https://github.com/opencv/opencv_contrib.git \
        && cd opencv_contrib \
        && git checkout $opencv_version \
        && cd .. \
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
        -D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib/modules \
        .. \
        && make install/strip \
        && cd /home \
        && rm -rf opencv opencv_contrib

WORKDIR /home

COPY Sample-Code-Cpp-Unix /home/sampleCode

CMD ["bash"]

FROM ubuntu:18.04

USER root

# Custom packages for some blog posts:

# Sample                | Packages
# ----------------------+-------------------------------------
# OCR                   | libtesseract-dev, tesseract-ocr-eng
# barcode-QRcodeScanner | libzbar-dev
# qt-test               | qt5-default

RUN export DEBIAN_FRONTEND noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        make \
        gcc \
        git \
        wget \
        libglib2.0-0 \
        libgtk2.0-dev \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
        libeigen3-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        pkg-config \
        libavformat-dev \
        libswscale-dev \
        libavcodec-dev \
        libavformat-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libtesseract-dev \
        tesseract-ocr-eng \
        libzbar-dev \
        qt5-default \
        && \
    apt-get clean

RUN wget -q -O /tmp/opencv.tar.gz https://codeload.github.com/opencv/opencv/tar.gz/4.4.0 && \
    cd /tmp/ && tar -xf /tmp/opencv.tar.gz && \
    wget -q -O /tmp/opencv_contrib.tar.gz https://codeload.github.com/opencv/opencv_contrib/tar.gz/4.4.0 && \
    cd /tmp/ && tar -xf /tmp/opencv_contrib.tar.gz && \
    mkdir /tmp/build && cd /tmp/build && \
    cmake -DBUILD_TESTS=OFF -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.4.0/modules ../opencv-4.4.0/ && \
    make -j4 && make install && \
    rm -rf /tmp/build && rm -rf /tmp/opencv*

RUN useradd ci -m -s /bin/bash -G users
USER ci

CMD bash

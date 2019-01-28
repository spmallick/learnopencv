# Text-Detection-DNN
Text detection using OpenCV DNN

## Getting the EAST Model

1. The `text detection` scripts use **EAST Model** which can be downloaded using this link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1

2. Once the file has been downloaded (~85 MB), unzip it using `tar -xvzf frozen_east_text_detection.tar.gz`.

3. After unzipping, copy the **`.pb`** model file to the working directory.

## Using the C++ code

### Compilation

To compile the **`text_detection.cpp`**, use the following:

```
cmake .
make
```

### Usage

Refer to the following to use the compiled file:

```
./textDetection --input=<input image path>
```

## Using the Python code

### Usage

Refer to the following to use the Python script:

```
python text_detection.py --input <image_path>
```

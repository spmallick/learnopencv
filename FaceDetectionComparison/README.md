There are 10 code files in this repo - 5 for C++ and 5 for Python:

1. face_detection_opencv_haar.cpp and face_detection_opencv_haar.py - For Haar based face detection
2. face_detection_opencv_dnn.cpp and face_detection_opencv_dnn.py - For OpenCV DNN based face detection
3. face_detection_dlib_hog.cpp and face_detection_dlib_hog.py - for dlib hog based face detection
4. face_detection_dlib_mmod.cpp and face_detection_dlib_mmod.py - for dlib mmod based face detection
5. run-all.cpp and run-all.py - for running all the 4 together

First of all Unzip the dlib.zip file

## For C++

**Compile**

Add path to the properly build OpenCV with DNN GPU Support and your CUDA:

```
cmake -D OpenCV_DIR=~/opencv -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ .
make
```

## For Python

_Note_: Our code is tested using Python 3.7.5, but it should also work with any other python3.x.

Install virtualenv:

```
pip install virtualenv
```

Create new virtual environment `env` and activate it:

```
python -m venv env
source  env/bin/activate
```

Install numpy:

```
pip install numpy
```

Install dlib:

```
pip install dlib
```

Create symlink to the properly build OpenCV with DNN GPU Support:

```
cd env/lib/python3.x/site-packages/
ln -s ~/opencv/build/cv2.so cv2.so
```

## Run

**If you don't pass any filename, it will use the web cam**

### For individual face detectors

**C++**

```
./face_detection_XXXX <filename>
```

_Note:_ for `face_detection_opencv_dnn.cpp` you can pass up to 3 arguments:

- video filename, if you'd like to run inference on a video instead of a camera:

```
./face_detection_opencv_dnn.out <filename>
```

- device, if you want to use CPU instead of CPU:

```
./face_detection_opencv_dnn.out "" cpu
```

- framework to specify Caffe (caffe) or TensorFlow (tf) network to use. Caffe network is set by default:

```
./face_detection_opencv_dnn.out "" gpu tf
```

**Python**

```
python face_detection_XXXX.py -video <filename>
```

_Note:_ for `face_detection_opencv_dnn.py` you can pass up to 3 arguments:

- filename, if you'd like to run inference on a video instead of a camera:

```
python face_detection_opencv_dnn.out --video <filename>
```

- device, if you want to use CPU instead of GPU:

```
python face_detection_opencv_dnn.out --video <filename> --device cpu
```

- framework to specify Caffe (caffe) or TensorFlow (tf) network to use. Caffe network is set by default:

```
python face_detection_opencv_dnn.out --video <filename> --device cpu --framework tf
```

### For running all together

**C++** ./run-all <filename>

**Python** python run-all.py --video <filename>

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

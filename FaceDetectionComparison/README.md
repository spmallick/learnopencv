There are 10 code files in this repo - 5 for C++ and 5 for Python
1. face_detection_opencv_haar.cpp and face_detection_opencv_haar.py - For Haar based face detection
2. face_detection_opencv_dnn.cpp and face_detection_opencv_dnn.py - For OpenCV DNN based face detection
3. face_detection_dlib_hog.cpp and face_detection_dlib_hog.py - for dlib hog based face detection
4. face_detection_dlib_mmod.cpp and face_detection_dlib_mmod.py - for dlib mmod based face detection
5. run-all.cpp and run-all.py - for running all the 4 together


First of all Unzip the dlib.zip file

## For C++
**Compile**
cmake .
make

## Run
**If you dont give any filename, it will use the webcam**

### For individual face detectors
**C++**
./face_detection_XXXX <filename>

**Python**
python face_detection_XXXX.py <filename>

### For running all together
**C++**
./run-all <filename>

**Python**
run-all.py <filename>


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

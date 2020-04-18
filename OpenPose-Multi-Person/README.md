The code belongs to the blog https://www.learnopencv.com/multi-person-pose-estimation-in-opencv-using-openpose

A.Requirements : 
1. OpenCV > 3.4.1
2. Matplotlib for Notebook
3. RUN getModels.sh from command line Or Download caffe model from http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel and put it in pose/coco folder


B.Compiling cpp file

Using g++
Command to compile the cpp file in ubuntu:

g++ -o3 -std=c++11 multi-person-openpose.cpp `pkg-config --libs --cflags opencv` -lpthread -o multi-person-openpose

Using CMake
cmake .
make

C. Usage
1. Python

python multi-person-openpose.py

2. C++

./multi-person-openpose


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>



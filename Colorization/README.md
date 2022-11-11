# CNN based Image Colorization using OpenCV
This repository contains code for the blog post [Convolutional Neural Network, CNN based Image Colorization using OpenCV](https://learnopencv.com/convolutional-neural-network-based-image-colorization-using-opencv/).

<img src="https://learnopencv.com/wp-content/uploads/2018/07/colorization-example-1024x414.png" alt="CNN based Image Colorization" width="900">

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/oy70bb6ntgikapcczq487/h?dl=1&rlkey=suwfba3j7c52u37aqq7dfvw7i)

## Usage

Run the getModels.sh file from command line to download the needed model files

	sudo chmod a+x getModels.sh
	./getModels.sh

Python:
Commandline usage to colorize 
a single image:
	python3 colorizeImage.py --input greyscaleImage.png
a video file:
	python3 colorizeVideo.py --input greyscaleVideo.mp4


C++:

## Compilation examples
g++ `pkg-config --cflags --libs opencv4` colorizeImage.cpp -o colorizeImage.out -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -std=c++11 
g++ `pkg-config --cflags --libs opencv4` colorizeVideo.cpp -o colorizeVideo.out -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -std=c++11 


## Commandline usage to colorize 
a single image:
	./colorizeImage.out greyscaleImage.png
a video file:
	./colorizeVideo.out greyscaleVideo.mp4


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

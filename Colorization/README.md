
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

Compilation examples:
g++ `pkg-config --cflags --libs opencv4` colorizeImage.cpp -o colorizeImage.out -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -std=c++11 
g++ `pkg-config --cflags --libs opencv4` colorizeVideo.cpp -o colorizeVideo.out -lopencv_core -lopencv_dnn -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -std=c++11 


Commandline usage to colorize 
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

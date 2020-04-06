
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
g++ -ggdb `pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.4.2/lib/pkgconfig/opencv.pc` colorizeImage.cpp -o colorizeImage.out
g++ -ggdb `pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.4.2/lib/pkgconfig/opencv.pc` colorizeVideo.cpp -o colorizeVideo.out

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

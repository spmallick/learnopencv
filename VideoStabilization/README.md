# Video Stabilization Using Point Feature Matching in OpenCV

**This repository contains the code for [Video Stabilization Using Point Feature Matching in OpenCV](https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/bllz75vkqg4adfc/AABgVD7PFSDV5TDunImK2diea?dl=1)


Please download input video from [here](https://drive.google.com/file/d/1l-dFUMD4Q9CzCbRuqYp0DIMjdFICJQT0/view?usp=sharing). Please make sure it is present in the directory from which the code is run.

# Run Code 
The code supports OpenCV 3.x and 4.x. 

## Python 
The code is tested on Python 3 only. 

```
python3 video_stabilization.py
```
By default it writes `video_out.mp4`.

## C++ 
Compile using the following
```
g++ -O3 -std=c++11 `pkg-config --cflags --libs opencv` video_stabilization.cpp -o video_stabilization
```
Run using the following command 
```
./video_stabilization
```
By default it writes `video_out.mp4`.
The code can also be compiled using **cmake** as follows:

```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

The executable file generated can be run using the following command

```
./video_stabilization
```


# Computer Vision & AI Consulting

If you need help implementing your computer vision or AI project, we provide consulting services at [BigVision.AI](https://bigvision.ai). 

Contact us at [contact@bigvision.ai](mailto:contact@bigvision.ai).

[![BigVision.AI](https://bigvision.ai/wp-content/uploads/2022/01/logo.png)](https://bigvision.ai)

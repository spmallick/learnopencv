## How to use OpenCV DNN Module with Nvidia GPU on Windows

**This repository contains the code for [How to use OpenCV DNN Module with Nvidia GPU on Windows](https://www.learnopencv.com/how-to-use-opencv-dnn-module-with-nvidia-gpu-on-windows) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/diefx6xylzn8ip4/AABCxrfLKwrCV65nRLQqblNLa?dl=1)

## Models

Download models from the following sources.

[COCO](https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel?dl=1)

[MPI](https://www.dropbox.com/s/drumc6dzllfed16/pose_iter_160000.caffemodel?dl=1)

## Run Code:

### C++
```
mkdir build
cd build
cmake -G "Visual Studio 16 2019" ..
cmake --build . --config Release
cd ..
./build/Release/OpenPoseVideo <input_file> <gpu/cpu>
```

### Python
```
python OpenPoseVideo.py --device=<cpu/gpu> 
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

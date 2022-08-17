
# How to use OpenCV DNN Module with NVIDIA GPUs on Linux

**This repository contains the code for [How to use OpenCV DNN Module with NVIDIA GPUs On Linux](https://www.learnopencv.com/opencv-dnn-with-gpu-support/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/ifwvza21tc0t7ma/AADuY0w1PgwDVmSLcwyQDhxfa?dl=1)

## Models

Download models from the following sources.

[COCO](https://www.dropbox.com/s/2h2bv29a130sgrk/pose_iter_440000.caffemodel?dl=1)

[MPI](https://www.dropbox.com/s/drumc6dzllfed16/pose_iter_160000.caffemodel?dl=1)

## Run Code:

### C++
```
mkdir build
cd build
cmake ..
make
cd ..
./build/OpenPoseVideo <input_file> <gpu/cpu>
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

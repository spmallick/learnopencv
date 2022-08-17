# Making a Low-Cost Stereo Camera Using OpenCV

**This repository contains code for [Making a low-cost stereo camera using OpenCV](https://www.learnopencv.com/making-a-low-cost-stereo-camera-using-opencv/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/7vs36qm9ntx5m7c/AAB88GWgDoCnNpJ4EAU3cRIra?dl=1)

## Using the C++ code
### Compilation
To compile the `calibrate.cpp`, `capture_images.cpp` and `movie3d.cpp` code files, use the following:
```shell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```
## Usage
Refer to the following to use the compiled files:

```shell
./build/capture_images
./build/calibrate
./build/movie3d
```

### Using the python code

Refer to the following to use the `calibrate.py`, `capture_images.py` and `movie3d.py` files respectively:

```shell
python3 calibrate.py
python3 capture_images.py
python3 movie3d.py
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

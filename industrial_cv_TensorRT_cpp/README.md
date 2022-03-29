# TensorRT C++ API
AUTOSAR C++ compliant code using TensorRT for deep learning inference

## Environment
All code was tested on Jetson AGX Xavier 16 GB Developer Kit running the latest **JetPack 4.6 (rev 3)** at the time of writing.

Kernel version:
```Shell
agx@agx-desktop:~/agxnvme/pyTensorRT$ uname -r
4.9.253-tegra
```

PyTorch is needed for generating onnx model using Torchvision
Please refer to our [python tutorial](https://github.com/spmallick/learnopencv/tree/master/industrial_cv_TensorRT_python) and use `segmodel_to_onnx.py` to generate onnx file and save it as `segmodel.onnx` in the `build` directory.

Tested PyTorch and torchvision versions
```Python
>>> import torch
>>> import torchvision
>>> torch.__version__
'1.9.0'
>>> torchvision.__version__
'0.10.0'
```

Use the [official guide](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048) to install PyTorch 1.9 if necessary.


## How to use

### Step 1
Convert the torchvision model to onnx format with
```Shell
#First go to the code for python tutorial and grab the onnx file
cd {path-where-cloned}/learnopencv/industrial_cv_TensorRT_python

python3 segmodel_to_onnx.py
#this will generate the onnx file
```

## Step 2
Ramp up the frequency of the GPU on the Jetson
```Shell
sudo su #type password
echo 1377000000 > /sys/devices/gpu.0/devfreq/17000000.gv11b/min_freq
#set minimum frequency to 1.4 GHz, max supported by Jetson AGX
exit #exit superuser mode
```

## Step 3
Move onnx file, Compile and run C++ code
```Shell
cd {path-where-cloned}/learnopencv/industrial_cv_TensorRT_cpp
mkdir build
cp ../industrial_cv_TensorRT_python/segmodel.onnx ./build
#copy onnx file to build directory

cd build
cmake -DCMAKE_BUILD_TYPE=Debug ../
make
./trt_test ./segmodel.onnx
#this will reproduce the fps numbers from python tutorial for FP16
```

## Results
These are same as what we got with python API. Refer to our TensorRT [python tutorial](https://github.com/spmallick/learnopencv/tree/master/industrial_cv_TensorRT_python) for details.

![Results](https://github.com/spmallick/learnopencv/blob/master/industrial_cv_TensorRT_python/TensorRT%20GPU%2C%20DLA%2C%20int8%20inference.png)

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>


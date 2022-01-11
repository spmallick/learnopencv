# pyTensorRT
Python tutorial for TensorRT

## Environment
All code was tested on Jetson AGX Xavier 16 GB Developer Kit running the latest **JetPack 4.6 (rev 3)** at the time of writing.

Kernel version:
```Shell
agx@agx-desktop:~/agxnvme/pyTensorRT$ uname -r
4.9.253-tegra
```
Tested TensorRT, PyTorch and torchvision versions
```Python
>>> import tensorrt as trt
>>> import torch
>>> import torchvision
>>> trt.__version__
'8.0.1.6'
>>> torch.__version__
'1.9.0'
>>> torchvision.__version__
'0.10.0'
```

The only dependency not installed by JetPack is PyTorch 1.9 along with its companion library torchvision which may be installed by following the [official guide](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).


## How to use

### Step 1
Convert the torchvision model to onnx format with
```Shell
python3 segmodel_to_onnx.py
```

## Step 2
Ramp up the frequency of the GPU on the Jetson
```Shell
sudo su #type password
echo 1377000000 > /sys/devices/gpu.0/devfreq/17000000.gv11b/min_freq
#set minimum frequency to 1.4 GHz, max supported by Jetson AGX
exit #exit superuser mode
```

Download MS COCO validation data (used for INT8 calibration)
```Shell
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

## Step 3
Run TensorRT inference, specifying precision (default: FP16), device (default: GPU) and video file (required)
```Shell
python3 pytrt.py --precision fp32 --device gpu --infile input.mp4 #FP32, GPU <-- baseline
python3 pytrt.py --precision fp16 --device gpu --infile input.mp4 #FP16, GPU
python3 pytrt.py --precision int8 --device gpu --infile input.mp4 #INT8, GPU
python3 pytrt.py --precision fp16 --device dla --infile input.mp4 #FP16, DLA
python3 pytrt.py --precision int8 --device dla --infile input.mp4 #INT8, DLA

python3 pytrt.py --precision int8 --device 2DLAs --infile input.mp4 
#INT8 async on 2 DLAs with minimum usage of GPU
```

## Results
![Results](https://github.com/spmallick/learnopencv/blob/master/industrial_cv_TensorRT_python/TensorRT%20GPU%2C%20DLA%2C%20int8%20inference.png)

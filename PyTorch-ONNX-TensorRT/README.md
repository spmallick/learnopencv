# How to convert a model from PyTorch to TensorRT and speed up inference
The blog post is here: https://www.learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/

To run PyTorch part:
```shell script
python3 -m pip install -r requirements.txt
python3 pytorch_model.py
```

To run TensorRT part:
1. Download and install NVIDIA CUDA 10.0 or later following by official instruction: [link](https://developer.nvidia.com/cuda-10.0-download-archive)
2. Download and extract CuDNN library for your CUDA version (login required): [link](https://developer.nvidia.com/rdp/cudnn-download)
3. Download and extract NVIDIA TensorRT library for your CUDA version (login required): 
[link](https://developer.nvidia.com/nvidia-tensorrt-6x-download). 
The minimum required version is 6.0.1.5. 
Please follow the [Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for your system and don't forget to install Python's part
4. Add the absolute path to CUDA, TensorRT, CuDNN libs to the environment variable ```PATH``` or ```LD_LIBRARY_PATH``` 
5. Install [PyCUDA](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda)

```shell script
python3 trt_inference.py
```

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
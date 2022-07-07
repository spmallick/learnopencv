# Demystifying GPU architectures for deep learning

This folder contains code for the blog post [Demystifying GPU architectures for deep learning](add-link)

## Environment
All code was tested on a PC with RTX 3090 and AMD Ryzen 5800X.

Kernel version:
```Shell
sf@trantor:~/Downloads$ uname -r
5.4.0-121-generic
```
## Download Code

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download.png" alt="Download Code" width="150">](https://www.dropbox.com/sh/b5y85yjyt1cxizn/AACpsOeqXcLJUMclEql7qXiEa?dl=1)

## How to use

## Compile and run

```Shell
nvcc cuda_matmul.cu -lm -o cu_mm.out
./cu_mm.out 2048 256 512
```

## Results

On the tested system, the GPU was about 650 times faster than the CPU.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>


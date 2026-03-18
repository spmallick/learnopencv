
# Deep Convolutional GAN in TensorFlow and PyTorch
**This repository contains code for [Deep Convolutional GAN in TensorFlow and PyTorch](https://learnopencv.com/deep-convolutional-gan-in-pytorch-and-tensorflow/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/hmlrgfz4wvpm6wf/AAChCpEYJK74EQxjs8PTHyIta?dl=1)


## Package Dependencies

The repository trains the Deep Convolutional GAN in both PyTorch and TensorFlow on the [Anime Faces dataset](https://github.com/bchao1/Anime-Face-Dataset). The current requirements have been validated with Python 3.12.x.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The PyTorch script now falls back to CPU automatically when the installed CUDA build does not support the available GPU.

## Directory Structure


```
├── PyTorch
│   ├── DCGAN_Anime_Pytorch.ipynb
│   └── dcgan_anime_pytorch.py
└── TensorFlow
    ├── DCGAN_Anime_Tensorflow.ipynb
    └── dcgan_anime_tesnorflow.py
```



## Instructions

### PyTorch

To train the Deep Convolutional GAN with PyTorch, go into the `PyTorch` folder and execute either the Python script or the notebook.

### TensorFlow

To train the Deep Convolutional GAN with TensorFlow, go into the `TensorFlow` folder and execute the notebook or script.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)

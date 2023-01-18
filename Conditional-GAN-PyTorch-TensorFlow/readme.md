# Conditional GAN in TensorFlow and PyTorch
This repository **This repository contains code for [Conditional GAN in TensorFlow and PyTorch](https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/) blogpost**.

## Package Dependencies
[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/wsmi6eo4bzqyat0/AABM7LHduc8tU55j7CR6kQE-a?dl=1)


This repository also trains the Conditional GAN in both Pytorch and Tensorflow on the Fashion MNIST and Rock-Paper-Scissors dataset. It is tested with the following CUDA versions.

- `Cuda-11.1`
- `Cudnn-8.0`

The Pytorch and Tensorflow scripts require [numpy](https://numpy.org/), [tensorflow](https://www.tensorflow.org/install), [torch](https://pypi.org/project/torch/).  To get the versions of these packages you need for the program, use pip. (Make sure pip is upgraded: ` python3 -m pip install -U pip`).

```
pip3 install -r requirements.txt 
```

## Directory Structure

```
├── PyTorch
│   ├── CGAN-PyTorch.ipynb
│   └── cgan_pytorch.py
└── TensorFlow
    ├── CGAN-FashionMnist-TensorFlow.ipynb
    ├── cgan_fashionmnist_tensorflow.py
    ├── CGAN-RockPaperScissor-TensorFlow.ipynb
    └── cgan_rockpaperscissor_tensorflow.py
```

## Instructions

### PyTorch

To train the Conditional GAN with Pytorch, please go into the `Pytorch` folder and execute the Jupyter Notebook.

### TensorFlow

To train the Conditional GAN with TensorFlow, please go into the `Tensorflow` folder and execute the Jupyter Notebook.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)

# Conditional GAN in TensorFlow and PyTorch
This repository **This repository contains code for [Conditional GAN in TensorFlow and PyTorch](https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/) blogpost**.

## Package Dependencies
[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/wsmi6eo4bzqyat0/AABM7LHduc8tU55j7CR6kQE-a?dl=1)


This repository trains Conditional GANs in both PyTorch and TensorFlow on the Fashion-MNIST and Rock-Paper-Scissors datasets. The current requirements have been validated with Python 3.12.x.

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

To train the Conditional GAN with PyTorch, go into the `PyTorch` folder and execute either the Python script or the notebook.

### TensorFlow

To train the Conditional GAN with TensorFlow, go into the `TensorFlow` folder and execute the notebook or script.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)

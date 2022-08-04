## Pix2Pix: Image-to-Image Translation with GAN in PyTorch & TensorFlow

**This repository contains code for [Pix2Pix: Image-to-Image Translation with GAN in PyTorch & TensorFlow](https://learnopencv.com/paired-image-to-image-translation-pix2pix/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/bgs8ifp334zuhxu/AADSkegkecS4borKaOi3W8oIa?dl=1)

## Package Dependencies

The repository trains the Pix2Pix GAN in both Pytorch and Tensorflow on the Edges2Shoes dataset. It is tested with:

- `Cuda-11.1`
- `Cudnn-8.0`

The Pytorch and Tensorflow scripts require [numpy](https://numpy.org/), [tensorflow](https://www.tensorflow.org/install), [torch](https://pypi.org/project/torch/). To get the versions of these packages you need for the program, use pip: (Make sure pip is upgraded: `python3 -m pip install -U pip`)

```
pip3 install -r requirements.txt 
```

## Directory Structure

```
├── PyTorch
│   ├── torch
│   │   ├── images
│   │   └── training_weights
│   ├── pix2pix_pytorch.ipynb
│   └── pix2pix_pytorch.py
├── TensorFlow
│   ├── model_single
│   ├── results_images_single
│   ├── pix2pix-tensorflow-multi_gpu.py
│   ├── pix2pix-tensorflow_single_gpu.ipynb
│   └── pix2pix-tensorflow-single_gpu.py
```

## Instructions

### PyTorch

To train the Pix2Pix GAN with Pytorch, please go into the `PyTorch` folder and execute the Jupyter Notebook.

### TensorFlow

To train the Pix2Pix GAN with TensorFlow, please go into the `TensorFlow` folder and execute the Jupyter Notebook.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/18c5719ef10afe9607af3e87e990068c942ae4cba8bd4d72d21950d6213ea97e/68747470733a2f2f7777772e6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30342f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)

## Pix2Pix: Image-to-Image Translation with GAN in PyTorch & TensorFlow

**This repository contains code for [Pix2Pix: Image-to-Image Translation with GAN in PyTorch & TensorFlow](https://learnopencv.com/paired-image-to-image-translation-pix2pix/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/bgs8ifp334zuhxu/AADSkegkecS4borKaOi3W8oIa?dl=1)

## Package Dependencies

The repository trains the Pix2Pix GAN in both PyTorch and TensorFlow on the Edges2Shoes dataset. The current requirements have been validated with Python 3.12.x.

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

To train the Pix2Pix GAN with PyTorch, go into the `PyTorch` folder and execute either the Python script or the notebook.

### TensorFlow

To train the Pix2Pix GAN with TensorFlow, go into the `TensorFlow` folder and execute the notebook or script.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)

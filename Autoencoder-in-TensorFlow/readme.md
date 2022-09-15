# Introduction to Autoencoder in TensorFlow

**This repsitory contains code for [Autoencoder in TensorFlow 2: Beginner’s Guide](https://learnopencv.com/autoencoder-in-tensorflow-2-beginners-guide/) blogpost**.

## Package Dependencies

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/vu2yg2qzfwogi8h/AABCHs4txiXMxOle_LFNSzMUa?dl=1)


This repository also trains the Autoencoder in Tensorflow framework on Fashion-MNIST and Cartoon dataset. The cartoon dataset can be downloaded from [here](https://google.github.io/cartoonset/).

The code is tested with:

- `Cuda-11.1`
- `Cudnn-8.0`

The Tensorflow notebook requires [numpy](https://numpy.org/), [tf-nightly-gpu](https://pypi.org/project/tf-nightly-gpu/), [opencv](https://pypi.org/project/opencv-python/), [sklearn](https://pypi.org/project/scikit-learn/). 
To get the versions of these packages you need for the program, use pip: (Make sure pip is upgraded: ` python3 -m pip install -U pip`)

```python
pip3 install -r requirements.txt 
```

## Add Virtualenv as Python Kernel in Jupyterlab

- Activate the virtualenv

```python
$ source your-venv/bin/activate
```

- Add the virtualenv as a jupyter kernel

```python
(your-venv)$ ipython kernel install --name "local-venv" --user
```

Replace `local-venv` with your virtualenv name.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/18c5719ef10afe9607af3e87e990068c942ae4cba8bd4d72d21950d6213ea97e/68747470733a2f2f7777772e6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30342f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)

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

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)

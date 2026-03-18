# Introduction to Autoencoder in TensorFlow

**This repsitory contains code for [Autoencoder in TensorFlow 2: Beginner’s Guide](https://learnopencv.com/autoencoder-in-tensorflow-2-beginners-guide/) blogpost**.

## Package Dependencies

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/vu2yg2qzfwogi8h/AABCHs4txiXMxOle_LFNSzMUa?dl=1)


This repository also trains the Autoencoder in Tensorflow framework on Fashion-MNIST and Cartoon dataset. The cartoon dataset can be downloaded from [here](https://google.github.io/cartoonset/).

The current requirements have been validated with Python 3.12.x. Create a fresh virtual environment and install the pinned dependencies before running the notebook locally.

```python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
pip install -r requirements.txt
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

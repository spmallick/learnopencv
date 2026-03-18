# TensorFlow Lite: Model Optimization for On-Device Machine Learning

**This repository contains code for [TensorFlow Lite: Model Optimization for On-Device Machine Learning](https://learnopencv.com/tensorflow-lite-model-optimization-for-on-device-machine-learning) blogpost**.


<img src="https://learnopencv.com/wp-content/uploads/2022/05/tflite_feature_image-1-scaled.jpg" align="middle">

## Install Requirements

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/6wjtg7edkdyrv1a/AAAwqeIq_4NZtMK_MoP5l00Ja?dl=1)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Tested with Python 3.12.x. The notebook now enables `TF_USE_LEGACY_KERAS=1` before importing TensorFlow so that `tensorflow-model-optimization==0.8.0` works correctly with TensorFlow 2.16. If you run the code outside the notebook, set that environment variable before importing TensorFlow.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

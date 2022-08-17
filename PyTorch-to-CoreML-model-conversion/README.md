# PyTorch to CoreML model conversion

**This repository contains the code for [PyTorch to CoreML model conversion](https://learnopencv.com/pytorch-to-coreml-model-conversion/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/flirha33aorrlrz/AADu-XLuR_BQFO5ZhgYNPaZIa?dl=1)

## Installation

All required libraries collected in the requirements.txt file. To create a new virtual environment and install the requirements, you can use these commands:

```
$ virtualenv -p python3.7 venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run the script

To convert any model from `torchvision.models` you can try our script by using this command:

```
$ python3 torch_to_coreml.py --model_name resnet18 --simplify
```

The `--model_name` argument should contain the name of any model from `torchvision.models`

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

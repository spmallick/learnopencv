# Efficient image loading

**This repository contains code for [Efficient image loading](https://www.learnopencv.com/efficient-image-loading/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/okyrb45okt8f5m9/AAACwY1iqwie4o8WhGHtHPdQa?dl=1)

## Local Setup

Tested with Python 3.12.x.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Example benchmark run:

```bash
python benchmark.py --path images --method cv2 pil lmdb tfrecords --mode RGB --iters 10
```

The benchmark now creates the LMDB output directory automatically and filters directory inputs down to real image files before loading them.

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

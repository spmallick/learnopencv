# YOLO26 Keypoint Estimation: Real-Time Pose Estimation with Ultralytics

**This repository contains the code for [YOLO26 Keypoint Estimation: Real-Time Pose Estimation with Ultralytics](https://learnopencv.com/yolo26-pose-estimation-tutorial/) blog post.**

![YOLO26 Keypoint Estimation](https://learnopencv.s3.us-west-2.amazonaws.com/wp-content/uploads/2026/04/16114728/play.png)
---

## Overview

This notebook demonstrates how to perform **real-time human pose estimation** using the **YOLO26** pose model from Ultralytics. YOLO26 builds on the YOLO family with several innovations targeted at keypoint estimation:

- **Residual Log-Likelihood Estimation (RLE)** for more accurate keypoint localization
- **End-to-end NMS-free inference** for predictable low-latency deployment
- **MuSGD optimizer** (SGD + Muon) for more stable training
- **Up to 43% faster CPU inference** compared to YOLO11
- **17 COCO keypoints** detected out of the box, plus support for custom non-human keypoints

## What's Covered

The notebook walks through pose estimation on **8 images** and **6 videos** spanning a range of real-world activities:

| Images | Videos |
|---|---|
| Yoga, Karate, Acro Yoga, Gym Workout, Walking, Gymnastics, Group Play, Sports Action | Boxing, Yoga, Parkour, Dance (side-by-side), Jump, Dance 2 (keyframe progression) |


## Model Variants

| Model | Params (M) | FLOPs (B) | mAP₅₀₋₉₅ (Pose) | T4 TensorRT (ms) |
|---|---:|---:|---:|---:|
| yolo26n-pose | 2.9 | 7.5 | 57.2 | 1.8 |
| yolo26s-pose | 10.4 | 23.9 | 63.0 | 2.7 |
| yolo26m-pose | 21.5 | 73.1 | 68.8 | 5.0 |
| yolo26l-pose | 25.9 | 91.3 | 70.4 | 6.5 |
| yolo26x-pose | 57.6 | 201.7 | 71.6 | 12.2 |

*Benchmarks on COCO Keypoints val2017 at 640×640.*

## How to Run

### Option 1 — Google Colab (recommended)


### Option 2 — Local Jupyter

```bash
git clone https://github.com/spmallick/learnopencv.git
cd learnopencv/YOLO26_Keypoint_Estimation
pip install ultralytics
jupyter notebook YOLO26_Keypoint_Estimation.ipynb
```

## Input Files

Sample inputs are distributed as a single zip in a public GitHub release:

```
https://github.com/spmallick/learnopencv/releases/download/YOLO26_Keypoint_Estimation/YOLO26_Keypoint_Estimation.zip
```

The notebook downloads and extracts it automatically into `./yolo26_workspace/inputs/`.

---

## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

<a href="https://opencv.org/courses/">
<p align="center">
<img src="https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

## Connect with us

- [Website](https://www.learnopencv.com/)
- [YouTube](https://www.youtube.com/@LearnOpenCV)
- [GitHub](https://github.com/spmallick/learnopencv)

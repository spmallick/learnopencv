# RF-DETR Segmentation: Complete Guide to Real-Time Instance Segmentation

This repository contains a comprehensive Jupyter notebook demonstrating **RF-DETR-Seg** — a state-of-the-art real-time instance segmentation model by [Roboflow](https://roboflow.com/), published at **ICLR 2026**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![RF-DETR](https://img.shields.io/badge/RF--DETR-Segmentation-purple)

## Overview

RF-DETR-Seg extends the RF-DETR detection framework with a lightweight segmentation head inspired by MaskDINO, adapted for the non-hierarchical DINOv2 ViT backbone. It achieves the best accuracy-latency trade-offs among real-time models on Microsoft COCO.

## What's Covered

- **Image Segmentation** — Single and batch instance segmentation with visualization
- **Video Segmentation** — Frame-by-frame processing pipeline for multiple videos
- **Webcam Segmentation** — Real-time inference using your webcam (local machines)
- **Advanced Visualizations** — Background blur, background replacement, heatmaps, color pop effects
- **Detection vs. Segmentation** — Side-by-side comparison
- **Deployment Tips** — Model selection guide, ONNX export, and optimization tips

## Model Zoo

| Model | Class Name | AP₅₀ | AP₅₀:₉₅ | Latency (ms) | Params (M) |
|-------|-----------|------|---------|-------------|-----------|
| Nano | `RFDETRSegNano` | 63.0 | 40.3 | 3.4 | 33.6 |
| Small | `RFDETRSegSmall` | 66.2 | 43.1 | 4.4 | 33.7 |
| Medium | `RFDETRSegMedium` | 68.4 | 45.3 | 5.9 | 35.7 |
| Large | `RFDETRSegLarge` | 70.5 | 47.1 | 8.8 | 36.2 |
| XLarge | `RFDETRSegXLarge` | 72.2 | 48.8 | 13.5 | 38.1 |
| 2XLarge | `RFDETRSeg2XLarge` | 73.1 | 49.9 | 21.8 | 38.6 |

> Latency measured on NVIDIA T4, TensorRT, FP16, batch size 1. All segmentation checkpoints are Apache 2.0 licensed.

## Getting Started

### Requirements

- Python >= 3.10
- GPU recommended (NVIDIA CUDA-compatible)

### Installation

```bash
pip install rfdetr supervision
```

For XLarge/2XLarge detection models:

```bash
pip install "rfdetr[plus]"
```

### Run the Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/RF_DETR_Segmentation.git
   cd RF_DETR_Segmentation
   ```

2. Open the notebook:

   ```bash
   jupyter notebook RF_DETR_Segmentation.ipynb
   ```

3. The notebook will automatically download sample images and videos from the [GitHub Release assets](https://github.com/spmallick/learnopencv/releases/tag/RF_DETR_Segmentation).


## Sample Results

The notebook processes 5 sample images and 5 sample videos:

**Images:** `Animals_1.png`, `Animals_2.png`, `City.png`, `Home.png`, `Street.png`

**Videos:** `BasketBall.mp4`, `Football.mp4`, `Horses.mp4`, `Kids_Playing.mp4`, `Street.mp4`


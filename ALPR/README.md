# Automatic License Plate Recognition

This repository contains code of our blog post Automatic License Plate Recognition. 

This code takes a two step approach where License plates are first detected using [YOLOv4](https://github.com/AlexeyAB/darknet) and OCR is then applied using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) on the detected License plates.

## Requirements
```
pip install -r requirements.txt
```
## Notebooks
This repository contains three jupyter notebooks, each one of performing different tasks.

- **[ALPR_inference.ipynb](https://colab.research.google.com/github/sanyam83/learnopencv/blob/master/ALPR/ALPR_inference.ipynb)**: This notebook contains the pipeline required for end to end inference of the Automatic License plate recognition on images and videos along with the implementation of tracker. Run this notebook to perform ALPR on an image or a video.

- **[License_plate_detection_YOLOv4.ipynb](https://colab.research.google.com/github/sanyam83/learnopencv/blob/master/ALPR/License_plate_detection_YOLOv4.ipynb)**: This notebook contains end to end implementation of license plate detection using YOLOv4. It includes code for training, evaluation and inference based on darknet.

- **[OCR_comparison.ipynb](https://colab.research.google.com/github/sanyam83/learnopencv/blob/master/ALPR/OCR_comparison.ipynb)**: This notebook compares performances of three of the PaddleOCR algorithms. The algorithms include, pp-ocr, pp-ocr(server), SRN.

## ALPR output example

![ALPR output](https://user-images.githubusercontent.com/64148610/158544760-75cee7a6-8461-4aba-b6a7-06b85723bc14.gif)

# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

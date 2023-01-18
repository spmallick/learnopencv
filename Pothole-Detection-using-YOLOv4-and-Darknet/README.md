# Pothole Detection using YOLOv4 and Darknet

**This repository contains the code for [Pothole Detection using YOLOv4 and Darknet](https://learnopencv.com/pothole-detection-using-yolov4-and-darknet/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/qlydlz7xlsjoq3a/AAAC1y4Ma0TnnGJR2mgNzgSia?dl=1)

**Here we train YOLOv4 and YOLOv4-Tiny models with 4 different experimental settings on a pothole detection dataset. We also run inference in real-time using the trained models.**

* The `jupyter_notebook` directory contains the Jupyter Notebook which will run end-to-end with one click. You can either run it locally if you have CUDA and cuDNN installed. Or you can upload the notebook to Colab and run it in a GPU enabled environment.

- The `custom_inference_script` directory contains the customized `darknet_video.py` Python file. The Darknet code has been customized to show the FPS on the videos when running the inference. After cloning and building Darknet, replace the original `darknet_video.py` file with the one in the `custom_inference_script` directory. 



***Download the YOLOv4 Pothole dataset trained weights [from here](https://drive.google.com/file/d/1vXTyWuvFCy7P0GEvQLYtpcxDi6yqZ9ce/view?usp=sharing).***



## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)
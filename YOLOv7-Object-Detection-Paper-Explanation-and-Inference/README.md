# YOLOv7 Object Detection - Paper Explanation and Inference

**This repository contains the code for [YOLOv7 Object Detection - Paper Explanation and Inference](https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/wzjudk2pem80vp6/AAASXVagcwMsB2DrdLXvwNO1a?dl=1)

In this blog post, we go through a comprehensive explanation of the YOLOv7 paper. This includes:

* The YOLOv7 architecture.
* The Bag of Freebies in YOLOv7.
* Training experiments and results from the paper.
* Running inference for object detection using the YOLOv7 and YOLOv7-Tiny model.
* Pose estimation using YOLOv7.



The `yolov7_keypoint.py` script contains the code for pose estimation using the `yolov7-w6-pose.pt` model. **Steps to run**:

* Clone the YOLOv7 repository.
* Download the `yolov7-w6-pose.pt` weights.
* Copy the script into the cloned `yolov7` repository/directory.
* Execute: `python yolov7_keypoint.py` 



![](results/video_4_keypoint_with_boxes.gif)

![](results/video_5_keypoint_with_boxes.gif)

## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/18c5719ef10afe9607af3e87e990068c942ae4cba8bd4d72d21950d6213ea97e/68747470733a2f2f7777772e6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30342f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)
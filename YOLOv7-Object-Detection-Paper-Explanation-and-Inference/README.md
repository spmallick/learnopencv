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

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)
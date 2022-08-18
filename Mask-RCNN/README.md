## Deep learning based Object Detection and Instance Segmentation using Mask RCNN in OpenCV (Python / C++)

**This repository contains code for [Deep learning based Object Detection and Instance Segmentation using Mask RCNN in OpenCV (Python / C++)](https://learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-rcnn-in-opencv-python-c/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/2u9g88r16g8te38/AABZBNbRkhSEIqzIvji9iamTa?dl=1)

**Python**

`wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz`
`tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz`

Download and extract the needed model files.

**Usage Examples :**

**Python**

`python3 mask_rcnn.py --image=cars.jpg`
`python3 mask_rcnn.py --video=cars.mp4`

It starts the webcam - if no argument provided.

**C++**

Compile using:

```g++ -ggdb `pkg-config --cflags --libs /Users/snayak/opencv/build/unix-install/opencv.pc` mask_rcnn.cpp -o mask_rcnn.out```

Run using:
`./mask_rcnn.out --image=cars.jpg`
`./mask_rcnn.out --video=cars.mp4`


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

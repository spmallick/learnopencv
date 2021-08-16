# Running OpenVino Models on Intel Integrated GPU



This repository contains:

* Python file to create the results JSON file for the COCO validation dataset.
* Juptyter notebook for calculating the mAP.



## Instructions

### Download the Video Used in the Post

* Download the video used in the post for inference from [this link](https://www.pexels.com/video/people-wearing-face-mask-in-public-area-4562551/).

### Getting the JSON Results File 

* To get the results JSON file for COCO validation set:

  * Execute `object_detection_demo_coco.py` by providing the correct path to the MS COCO validation dataset by editing the Python file.

  * Execute using the following commands:

    ```
    python object_detection_demo_coco.py --model fp16/frozen_darknet_yolov4_model.xml -at yolo -i mscoco/val2017 --loop -t 0.2 --no_show -r -nireq 4
    ```


### mAP Calculation

* Put the `pycocoEvalDemo.ipynb` in the `cocoapi/PythonAPI`.

* Run the `pycocoEvalDemo.ipynb` Notebook by providing the correct path the `results.json` 
* The correct path to the MS COCO evaluation JSON file also needs to be provided. Please check the path according to your directory structure of the MS COCO dataset.



# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://camo.githubusercontent.com/18c5719ef10afe9607af3e87e990068c942ae4cba8bd4d72d21950d6213ea97e/68747470733a2f2f7777772e6c6561726e6f70656e63762e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032302f30342f41492d436f75727365732d42792d4f70656e43562d4769746875622e706e67)](https://opencv.org/courses/)
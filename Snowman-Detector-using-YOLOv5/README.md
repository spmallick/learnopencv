# Snowman-Detector-using-YOLOv5



## Instructions to Run the Notebook

The notebook swill run end to end on Colab by doing **Run all**.

### For Training

* Make the `TRAIN = True` under the **Constant/Config Setup** heading. Can also choose the number of epochs here.
* If only wanting to run validation, make `TRAIN = False`. But for this already a trained model should be present.

### train_yolov5_snowman_small_train_all.ipynb

* This notebook trains a YOLOv5 small model entirely. All layers are trained.

### train_yolov5_snowman_medium_freeze_layers.ipynb

* This notebook trains a YOLOv5 medium model by freezing the first 21 layers and training the top few layers only.
* Additionally this notebook contains Weights and Biases logging and shows the ground truth images with bounding boxes.



# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
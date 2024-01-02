# Integrating ADAS with Keypoint Feature Pyramid Network for 3D LiDAR Object Detection

This repository contains the training scripts for Keypoint Feature Pyramid Network, specifically for 3D LiDAR Object Detection. In this case, the KITTI 360 Vision dataset has been used to train the detection model.   

It is part of the LearnOpenCV blog post - [Integrating ADAS with Keypoint Feature Pyramid Network for 3D LiDAR Object Detection](https://learnopencv.com/3d-lidar-object-detection/).

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://www.dropbox.com/scl/fi/3n1s68jtfkjmw2f5e5ctv/3D-LiDAR-Object-Detection.zip?rlkey=d8q6xvlxis4oxso4qki87omvc&dl=1)

![](readme_images/3D-LiDAR-Object-Detection.gif)

### Environment Setup

Run the following commands on a new terminal window for creating a new environment with the required packages: 

```shell script
cd SFA3D
pip install -r requirements.txt
```

### Dataset Visualization
To visualize 3D point clouds with 3-dimensional bounding boxes, run the following commends: 

```shell script
cd sfa/data_process
python kitti_dataset.py
```

### Inference
There is an instance of a pre-trained model in this repository. You can use it to run inference: 

```shell script
python test.py --gpu_idx 0 --peak_thresh 0.2
```

### Video Demonstration
Similarly, inference can be run on a video stream: 

```shell script
python demo_2_sides.py --gpu_idx 0 --peak_thresh 0.2
```
### Training Pipeline

##### Single Machine w/ Single GPU

```shell script
python train.py --gpu_idx 0
```

##### Single Machine w/ Multiple GPUs

```shell script
python train.py --multiprocessing-distributed --world-size 1 --rank 0 --batch_size 64 --num_workers 8
```

### Evaluation Metrics - TensorBoard
To track the training progress, go to `logs/` folder and run: 

```shell script
cd logs/<saved_fn>/tensorboard/
tensorboard --logdir=./
```

Then, just go to [http://localhost:6006/](http://localhost:6006/)


## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)
# Deep Learning based Object Detection using YOLOv3 with OpenCV
This repository contains code for the blog post [Deep Learning based Object Detection using YOLOv3 with OpenCV ( Python / C++ )](https://learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/).

<img src="https://learnopencv.com/wp-content/uploads/2018/08/nms-threshold-object-detection.gif" alt="YOLOv3 detections" width="900">

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/nkpnbcnm3sb5ng6ndm9jx/h?dl=1&rlkey=0bav8t0u8wygk39fsa9hvg9jh)


### Download the Models

Run the **getModels.sh** file from command line to download the needed model files

```bash
  sudo chmod a+x getModels.sh
  ./getModels.sh
```


### How to run the code

Command line usage for object detection using YOLOv3 

* Python

  * Using CPU

    * A single image:
    ```bash
    python3 object_detection_yolo.py --image=bird.jpg --device 'cpu'
    ```

    * A video file:
     ```bash
     python3 object_detection_yolo.py --video=run.mp4 --device 'cpu'
     ```

  * Using GPU

    * A single image:
    ```bash
    python3 object_detection_yolo.py --image=bird.jpg --device 'gpu'
    ```

    * A video file:
     ```bash
     python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'
     ```


* C++:

  * Using CPU

    * A single image:

    ```bash
    ./build/object_detection_yolo --image=bird.jpg --device=cpu
    ```

    * A video file:

    ```bash
     ./build/object_detection_yolo --video=run.mp4 --device=cpu
    ```

  * Using GPU

    * A single image:

    ```bash
    ./build/object_detection_yolo --image=bird.jpg --device=gpu
    ```

    * A video file:

    ```bash
     ./build/object_detection_yolo --video=run.mp4 --device=gpu
    ```


### Compilation examples

* Using g++
 
```bash
g++ -ggdb pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.4.2/lib/pkgconfig/opencv.pc object_detection_yolo.cpp -o object_detection_yolo.out
```

* Using CMake

  * On Unix systems

  ```bash
  mkdir build && cd build
  cmake ..
  cmake --build . --config Release
  cd ..
  ```

  * On Windows systems

  ```bash
  mkdir build
  cd build
  cmake -G "Visual Studio 16 2019" ..
  cmake --build . --config Release
  cd ..
  ```

**Note: To run on Windows system, change syntax accordingly:**

```bash
.\build\Release\object_detection_yolo --video=run.mp4 --device=gpu
```

### Results of YOLOv3
<img src = "https://github.com/gulshan-mittal/learnopencv/blob/dev1/ObjectDetection-YOLO/bird_yolo_out_py.jpg" width = 400 height = 300/>


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

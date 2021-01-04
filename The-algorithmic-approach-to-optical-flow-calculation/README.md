# Optical Flow demo

This contains the code for **Optical Flow in OpenCV (C++/Python)** post
For more information - visit 
[Optical Flow in OpenCV (C++/Python)](https://www.learnopencv.com/optical-flow-in-opencv)

## Python

### Installation

Before you start the demo of Optical Flow calculation, you need to create a virtual environment in your working
directory and install the required libraries:

```Shell
virtualenv -p python3.7 venv
source venv/bin/activate
pip install -r reqirements.txt
```

### Sparse Optical Flow

There is a demo `lucas_kanade.py` script of **Lucas-Kanade** algorithm which can be run with this command:

```
python3 demo.py --algorithm lucaskanade --video_path videos/car.mp4
```

### Dense Optical Flow

The wrapper of Dense Optical Flow algorithms `dense_optical_flow.py` can run a couple of OpenCV's algorithm
implementations:

- To start the **Dense Lucas-Kanade** algorithm:
  ```
  python3 demo.py --algorithm lucaskanade_dense --video_path videos/people.mp4
  ```
- To start the **Farneback** algorithm:
  ```
  python3 demo.py --algorithm farneback --video_path videos/people.mp4
  ```
- To start the **RLOF** algorithm:
  ```
  python3 demo.py --algorithm rlof --video_path videos/people.mp4
  ```

## C++

### Installation

Before you start the demo of Optical Flow calculation, you need to build the project:

```Shell
cd algorithms
cmake .
make
```

### Sparse Optical Flow

There is a demo `lucas_kanade.cpp` script of **Lucas-Kanade** algorithm which can be run with this command:

```
./OpticalFlow ../videos/car.mp4 lucaskanade
```

### Dense Optical Flow

The wrapper of Dense Optical Flow algorithms `dense_optical_flow.py` can run a couple of OpenCV's algorithm
implementations:

- To start the **Dense Lucas-Kanade** algorithm:
  ```
  ./OpticalFlow ../videos/car.mp4 lucaskanade_dense
  ```
- To start the **Farneback** algorithm:
  ```
  ./OpticalFlow ../videos/car.mp4 farneback
  ```
- To start the **RLOF** algorithm:
  ```
  ./OpticalFlow ../videos/car.mp4 rlof
  ```


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

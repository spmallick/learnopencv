## Introduction to ROS2

<img src="media/intro-to-ros2.gif">
<br>
<br>
<br>

**Blog: https://learnopencv.com/robot-operating-system-introduction**



ROS is a very common component in robotics and has many technical tutorials and resources available on the internet. However, through this blog, our objective is to provide a detailed understanding of the internal **workings of ROS2**, **how DDS works**, the **need for DDS**, the ROS1 middleware architecture, and the data flow in ROS2. Additionally, we discuss how to use this tool in Python, covering various topics such as packages, nodes, topics, publishers, subscribers, and services. At the end, for more hands-on understanding, we have created a capstone project where we **integrate Monocular SLAM with ROS2 using Python**. We hope this will serve as a beginner-friendly gateway for anyone who wants to learn ROS2 and get into robotics.


## How to Run:

Donwload the test video from [this link](https://www.dropbox.com/scl/fi/qsck7st5h85e3sniw0daq/test_ohio.mp4?rlkey=h8n5mf4aue0hocj44d4rzocbf&st=t3vdoc5a&dl=1) and place it inside `src/slam/resource/videos/`.


```bash
$ cd ../../ # come to the workspace home directory
$ colcon build # this will build all the packages inside src
$ source install/setup.bash # source the workspace
$ ros2 launch slam slam.launch.py
```
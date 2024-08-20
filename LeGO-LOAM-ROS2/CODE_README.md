# LeGO-LOAM-ROS2



This code is a fork from [LeGO-LOAM-SR](https://github.com/eperdices/LeGO-LOAM-SR) to migrate [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM) algorithm to ROS2 humble.

This code only done minor change to migrate [LeGO-LOAM-SR](https://github.com/eperdices/LeGO-LOAM-SR) from ROS2 dashing to ROS2 humble, since there are some syntax difference in [rclcpp](https://docs.ros2.org/humble/api/rclcpp/index.html) between dashing and humble.

This code does not modify and/or improve the original [LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM) algorithm.

## 1. About the original LeGO-LOAM

[LeGO-LOAM](https://github.com/RobustFieldAutonomyLab/LeGO-LOAM) contains code for a lightweight and ground optimized lidar odometry and mapping (LeGO-LOAM) system for ROS compatible UGVs.
The system takes in point cloud from a Velodyne VLP-16 Lidar (palced horizontal) and it outputs 6D pose estimation in real-time. A demonstration of the system can be found here -> https://www.youtube.com/watch?v=O3tz_ftHV48
<!--
[![Watch the video](/LeGO-LOAM/launch/demo.gif)](https://www.youtube.com/watch?v=O3tz_ftHV48)
-->
<p align='center'>
    <img src="/LeGO-LOAM/launch/demo.gif" alt="drawing" width="800"/>
</p>

## 2. Dependencies

- test in Ubuntu 20.04
- [ROS1 Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [ROS2 humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- [gtsam](https://github.com/borglab/gtsam/releases) (Georgia Tech Smoothing and Mapping library, 4.0.0-alpha2)
  ```
  wget -O ~/Downloads/gtsam.zip https://github.com/borglab/gtsam/archive/4.0.0-alpha2.zip
  cd ~/Downloads/ && unzip gtsam.zip -d ~/Downloads/
  cd ~/Downloads/gtsam-4.0.0-alpha2/
  mkdir build && cd build
  cmake ..
  sudo make install
  ```
- [colcon](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html)
  ```
  sudo apt install python3-colcon-common-extensions
  ```

## 3. Compile

You can use the following commands to download and compile the package.

```
mkdir -p ~/dev_ws/src
cd ~/dev_ws/src
git clone https://github.com/Thuniinae/LeGO-LOAM-ROS2.git
cd ..
colcon build
```

## 4. The system

LeGO-LOAM is speficifally optimized for a horizontally placed lidar on a ground vehicle. It assumes there is always a ground plane in the scan.

The package performs segmentation before feature extraction.

<p align='center'>
    <img src="/LeGO-LOAM/launch/seg-total.jpg" alt="drawing" width="400"/>
</p>

Lidar odometry performs two-step Levenberg Marquardt optimization to get 6D transformation.

<p align='center'>
    <img src="/LeGO-LOAM/launch/odometry.jpg" alt="drawing" width="400"/>
</p>

## 5. New sensor and configuration

To customize the behavior of the algorithm or to use a lidar different from VLP-16, edit the file **config/loam_config.yaml**.

One important thing to keep in mind is that our current implementation for range image projection is only suitable for sensors that have evenly distributed channels.
If you want to use our algorithm with Velodyne VLP-32c or HDL-64e, you need to write your own implementation for such projection.

If the point cloud is not projected properly, you will lose many points and performance.

**The IMU has been remove from the original LeGO-LOAM code.**

## 6. Run the package

System can be started using the following command:

```
echo source /opt/ros/noetic/setup.bash >> .bashrc
echo source /opt/ros/humble/setup.bash >> .bashrc
echo source ~/dev_ws/install/setup.bash >> .bashrc
```
open another terminal
```
ros2 launch lego_loam_sr run.launch.py
```
- for some unknown reason, launching the package right after sourcing ros and workspace in the same terminal may cause error.

Some sample bags can be downloaded from [here](https://github.com/RobustFieldAutonomyLab/jackal_dataset_20170608).

### 6.1 Run ROS1 rosbag

To use rosbags from ROS1, plugin [rosbag_v2](https://github.com/ros2/rosbag2_bag_v2) can be usefull.

Play rosbag using rosbag_v2:

```
source /opt/ros/noetic/setup.bash
source /opt/ros/humble/setup.bash
ros2 bag play --topics /velodyne_points -s rosbag_v2 <path_to_bagfile>
```
note:
    topic of this package subscribed is remapped from '/lidar_points' to '/velodune_points', can be change in launch/run.launch.py line 45 if needed.

### 6.2 Lidar data-set
This code had been test using [Stevens data-set](https://github.com/TixiaoShan/Stevens-VLP16-Dataset).

This dataset, [Stevens data-set](https://github.com/TixiaoShan/Stevens-VLP16-Dataset), is captured using a Velodyne VLP-16, which is mounted on an UGV - Clearpath Jackal, on Stevens Institute of Technology campus.
The VLP-16 rotation rate is set to 10Hz. This data-set features over 20K scans and many loop-closures.
The TF transform in the bags is provided by LeGO-LOAM (without enabling the loop-cloure function, pure lidar odometry). 

To use this dataset to test LeGO-LOAM-ROS2, only topic /velodyne_points should be played. Topic /tf contain same structure of trasformation as LeGO-LOAM-ROS2, playing it will cause interference.


<p align='center'>
    <img src="/LeGO-LOAM/launch/dataset-demo.gif" alt="drawing" width="600"/>
</p>
<p align='center'>
    <img src="/LeGO-LOAM/launch/google-earth.png" alt="drawing" width="600"/>  
</p>

## 7. Cite *LeGO-LOAM*

Thank you for citing our *LeGO-LOAM* paper if you use any of this code:
```
@inproceedings{legoloam2018,
  title={LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain},
  author={Tixiao Shan and Brendan Englot},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4758-4765},
  year={2018},
  organization={IEEE}
}
```

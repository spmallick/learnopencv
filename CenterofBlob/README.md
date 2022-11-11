# Center of Blob using Python and CPP

This repository contains code for the blog post [Find the Center of a Blob using OpenCV (C++/Python)](https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/).

<img src="https://learnopencv.com/wp-content/uploads/2018/07/single-blob-image-768x307.png" alt="Centre of Blob" width="900">


[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/eftkr2kxe0wrm6i386hcs/h?dl=1&rlkey=5iaq04xe7xvprktki6hvcndnx)


To run the code to find center of a single blob, run the following commands.

Python

`python3 single_blob.py --ipimage image_name`


C++

1. ``g++ single_blob.cpp `pkg-config opencv --cflags --libs` -o output``

2. `./output image_name`

To run the code to find center of multiple blobs, run the following commands:-

Python

`python3 center_of_multiple_blob.py --ipimage image_name`

C++

1. ``g++ center_of_multiple_blob.cpp `pkg-config opencv --cflags --libs` -o output``

2. `./output image_name`


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>


# Fast Image Downloader for Open Images V4

**This repository contains the code for [Fast Image Downloader for Open Images V4](https://learnopencv.com/fast-image-downloader-for-open-images-v4/) blog post**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/84co6f4d8sfia31/AACJ2td0E8hc40XIjTqg_Sgya?dl=1)

1. Install awscli

`sudo pip3 install awscli` 

2. Get the relevant OpenImages files needed to locate images of interest

`wget https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv`

`wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv`
  
 For windows: `python -m wget <URL>`

3. Download the images from OpenImagesV4

`python3 downloadOI.py --classes 'Ice_cream,Cookie' --mode train`

If there is a space in the class name, it should be replaced by an underscore(‘_’), e.g in the case of 'Ice cream' as shown above.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

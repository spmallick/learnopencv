# Face Reconstruction using EigenFaces

The repository contains code for the blog post [Face Reconstruction using EigenFaces](https://www.learnopencv.com/face-reconstruction-using-eigenfaces-cpp-python/).

<p align="center"><img src="https://learnopencv.com/wp-content/uploads/2018/01/face-reconstruction-using-eigenfaces.jpg" alt="EigenFaces" width="900"></p>

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/scl/fo/dwcfw2tg65dxt1ijbtnje/h?dl=1&rlkey=q3cc6nxe3t6gt76fdutr59trd)


## Steps to Train you own Model

1. Download [images](http://www.learnopencv.com/wp-content/uploads/2018/01/CalebA-1000-images.zip). These images are the first 1000 images of the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). You can create a larger model by using more [aligned and cropped images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0) from the CelebA dataset. 
2. Use **createPCAModel.cpp** or **createPCAModel.py** to create the modelfile **pcaParams.yml**.
3. Use **reconstructFace.cpp** or **reconstructFace.py** to reconstruct the face. It needs the **pcaParams.yml** file. 


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

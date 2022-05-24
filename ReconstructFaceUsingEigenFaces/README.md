Please see the following [blog post](https://www.learnopencv.com/face-reconstruction-using-eigenfaces-cpp-python/) for more details about this code

[Face Reconstruction using EigenFaces (C++/Python)](https://www.learnopencv.com/face-reconstruction-using-eigenfaces-cpp-python/)

To train your own model 
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

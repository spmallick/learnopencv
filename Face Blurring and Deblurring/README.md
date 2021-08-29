# Face Blurring + Deblurring Model

## **GOAL:**

In this project we will play with an image to detect the faces and then we will apply the blurr effect on some part of that image. And then we will also learn how to deblur the face. This project will be done with the help of some libraries like keras, tensorflow, opencv, matplotlib, numpy, scikit-image, mtcnn using Python programming.

**DATASET:**

We will be using image of our choice.


**LIBRARIES NEEDED:**
- Tensorflow
- Keras
- MTCNN
- Numpy
- OpenCV
- Matplotlib
- Scikit-image

## **What I Have Done:**

### **1. For Face Blurring**

![image](https://raw.githubusercontent.com/hrugved06/learnopencv/master/Face%20Blurring%20and%20Deblurring/images/assets/ppl.jpg)![image](https://raw.githubusercontent.com/hrugved06/learnopencv/master/Face%20Blurring%20and%20Deblurring/images/assets/arrow.png)![image](https://raw.githubusercontent.com/hrugved06/learnopencv/master/Face%20Blurring%20and%20Deblurring/images/assets/Blurred_Image.png)

I have created a project in which we will detect some faces and then will apply the blurr effect on them. 
First We will read/load the required images for for this project and then getting the coordinates for different parts of faces present in the image.

Then drawing a rectangle using X , Y axes and then displaying the face with a boundary box.

After getting the proper coordinates for all the faces in image we will apply the blurring effect using the height (h) and width (w) coordinates.

And then using the for loop we are blurring each face present in the image and finally saving & displaying the snapchat filtered image.

**WORKING**

To start using this project, follow the below guidelines: 


- Install the `requirements.txt` using command

```
pip install -r requirements.txt
```

- Run `face_detection_and_blurring.ipynb` file in Google Colab or Jupyter Notebook or any other platform 💻


### **2. Face deblurring operation**

![image](https://user-images.githubusercontent.com/78999467/115102986-2b809c80-9f6e-11eb-82f7-e6a5a5de3f85.png)![image](https://user-images.githubusercontent.com/78999467/115102966-04c26600-9f6e-11eb-841d-994a925343c6.png)![image](https://user-images.githubusercontent.com/78999467/115102989-33404100-9f6e-11eb-832e-91348d7a0b7c.png)

**Face deblurring operation** is the task of estimating a clear image from its degraded blur image and recovering the sharp contents and textures. The aim of **face deblurring** is to restore clear images with more explicit structure and facial details. The face deblurring problem has attracted considerable attention due to its wide range of applications.

**Image deblurring** is an ill-posed problem in computer vision. There has been a remarkable process in the direction of solving the blur kernel and the latent image alternately. The **CNN-based methods** are developed to solve the deblurring problem to restore the intermediate properties or the blur kernels. In addition, the framework which utilizes the end-to-end model for direct latent image prediction has also been proposed. 

We first propose an **end-to-end convolutional neural network model** to learn effective features from the blurred face images and then estimate a latent one. To constrain the network, we introduce to utilize a transfer learning framework to learn the multiple features. In addition, we adopt well-established deep networks to obtain extremely expressive features and achieve high-quality results.

**Domain-specific methods** for deblurring targeted object categories, e.g. text or faces, frequently outperform their generic counterparts, hence they are attracting an increasing amount of attention. In this work, we develop such a domain-specific method to tackle the deblurring of human faces, henceforth referred to as face deblurring. 

Studying faces is of tremendous significance in computer vision, however, face deblurring has yet to demonstrate some convincing results. This can be partly attributed to the combination of 

1.  poor texture and 
2.  highly structured shape that yields the contour/gradient priors (that are typically used) sub-optimal.

In our work instead of making assumptions over the prior, we adopt a **learning approach** by inserting weak supervision that exploits the well-documented structure of the face. Namely, we utilize a deep network to perform the deblurring and employ a face alignment technique to pre-process each face. We additionally surpass the requirement of the deep network for thousands of training samples, by introducing an efficient framework that allows the generation of a large dataset.

**Working**

1) Training an End-to-End model for deblurring of images (CelebA) following the work in CNN For Direct Text Deblurring, using Keras. The first layer filter size is adjusted to be approximately equal to the blur kernel size. Pre-Trained model with weights and some images from test set are uploaded.
2) **Importing Necessary Packages**
3) **Loading Images**

- Only showing a small set of images from the local test set we generated.

4) **Loading input blurred images:**
![image](https://user-images.githubusercontent.com/78999467/115102547-430a5600-9f6b-11eb-9691-74045164dbbc.png)

5) **Defining CNN Model for Training Model**

- The model has been trained on a much larger dataset of CelebA images.
- Loaded the weight file celebA_deblur_cnn_weights.h5

6) **Deblurred Faces**

- Deblurred images as output:

![image](https://user-images.githubusercontent.com/78999467/115102584-8fee2c80-9f6b-11eb-8db9-068dc3ab2ff6.png)




**CONCLUSION:**

Finally, we have created a project in which we have detected some faces and then applied the blurr effect on them. And we have also learnt how to work with such type of blurring effects and we can blurr any part of image in this way, the output we got with this project is also attached here.
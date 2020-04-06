## Character Classification (of Synthetic Dataset) using Keras (modified LeNet)

**Step 1:**

Download backgrounds and put the light and dark backgrounds separately. We'll be using them for creating synthetic dataset. We have uploaded sample backgrounds in light_backgrounds and dark_backgrounds for reference. 

**Step 2:**

Download fonts from [here](https://fonts.google.com/). These fonts will be used for randomly selected font-type while creating synthetic dataset. 

**Step 3:**

Create synthetic data using ImageMagick. We have given an intuition behind creating synthetic data, in our blog. This can be done with following command:

`python3 generate-images.py` 

The script first generates two directories *light_background_crops* and *dark_background_crops* containing 32x32 backgrounds crops. It then adds text and other artifacts like blur/noise/distortion to the backgrounds. To regenerate all data, delete *light_background_crops* and *dark_background_crops*. To generate training images, open the script and set OUTPUT_DIR = 'train/' and NUM_IMAGES_PER_CLASS = 800. Similarly, to generate test images, set OUTPUT_DIR = 'test/' and NUM_IMAGES_PER_CLASS = 200. 

**Step 4:** 

Training the model on the given dataset. A modified LeNet structure has been used to train our model, using Keras. This can be done with following command:

`python3 train_model.py`

**Step 5:**

In order to predict the digit or character in an image, execute the following command. Give the test image path as the argument. 

`python3 make_predictions.py <image_path>`


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>

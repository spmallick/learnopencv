## Character Classification (of Synthetic Dataset) using Keras (CNN)

**Step-1:**

Download backgrounds and put the light and dark backgrounds separately. We'll be using them for creating synthetic dataset. We have uploaded sample backgrounds in light_backgrounds and dark_backgrounds for reference. 

**Step-2:**

Download fonts from [here](https://fonts.google.com/). These fonts will be used for randomly selected font-type while creating synthetic dataset. 

**Step-3:**

Create synthetic data using ImageMagick. We have given an intuition behind creating synthetic data, in our blog. This can be done with following command:

`python3 generate-images.py` 

**Step-4:** 

Training the model on the given dataset. A modified LeNet structure has been used to train our model, using Keras. This can be done with following command:

`python3 train_model.py`

**Step-5:**

In order to predict the digit or character in an image, execute the following command. Give the test image path as the argument. 

`python3 make_predictions.py <image_path>`

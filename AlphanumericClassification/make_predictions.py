import cv2 # for reading,writing or showing image
import numpy as np
import matplotlib.pyplot as plt

# import necessary modules
import keras
from keras.preprocessing import image

import tensorflow as tf # For conversion of image to a tensor 
import time
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.models import load_model
# Specify model structure
def create_model():
    model = Sequential()
    
    # conv => relu => maxpool2d 
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64)) # FC Layer with 64 neurons
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # To overcome 'overfitting'
    
    model.add(Dense(36)) # number of classes = 36
    model.add(Activation('softmax')) 
    
    return model

def load_image(img_path, show=False):
    '''
    Function: Convert image to tensor
    Input: image_path (eg. /home/user/filename.jpg) 
        (Note prefer having absolute path)
           show (default = False), set if you want to visualize the image
    Return: tensor format of image
    '''
    # load image using image module
    # convert to (32, 32) - if not. 
    img = image.load_img(img_path, target_size=(32, 32))  # Path of test image
    # show the image if show=True
    if show:
        plt.imshow(img)                           
        plt.axis('off')
    
    # converting image to a tensor
    img_tensor = image.img_to_array(img)                  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    
    # return converted image
    return img_tensor

def predict(weights_path, image_path):
    '''
    Function: loads a trained model and predicts the class of given image
    Input: weights_path (.h5 file, prefer adding absolute path)
           image_path (image to predict, prefer adding absolute path)
    Returns: none
    '''
    model = create_model()
    model.load_weights(weights_path)
    image = load_image(image_path, show=True) # load image, rescale to 0 to 1
    class_ = model.predict(image) # predict the output, returns 36 length array
    print("Detected: ", class_[0]) # print what is predicted
    output_indice = -1 # set it initially to -1
    
    # get class index having maximum predicted score
    for i in range(36):
        if(i == 0):
            max = class_[0][i]
            output_indice = 0
        else:
            if(class_[0][i] > max):
                max = class_[0][i]
                output_indice = i
    
    # append 26 characters (A to Z) to list characters
    characters = []
    for i in range(65, 65+26):
        characters.append(chr(i))
    # if output indice > 9 (means characters)
    if(output_indice > 9):
        final_result = characters[(output_indice - 9) - 1]
        print("Predicted: ", final_result)
        print("value: ", max) # print predicted score
    # else it's a digit, print directly
    else:
        print("Predicted: ", output_indice)
        print("value: ", max) # print it's predicted score

predict("weights.h5", "image.jpg") # Specify weights file and Test image

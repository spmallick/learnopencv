import cv2
import numpy as np
import keras
from keras.preprocessing import image
import tensorflow as tf
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
from keras import optimizers
from keras.layers.convolutional import Conv2D


def create_model():   # Specify model structure
    model = Sequential()
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
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36))
    model.add(Activation('softmax'))
    
    return model

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(32, 32))  # Path of test image
    if show:
        plt.imshow(img)                           
        plt.axis('off')
        
        
    img_tensor = image.img_to_array(img)                  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    

    return img_tensor

def load_trained_model(weights_path, image_path):
    model = create_model()
    model.load_weights(weights_path)

    
    image = load_image(image_path, show=True) # load image, rescale to 0 to 1
    
    class_ = model.predict(image) # predict the output, returns 36 length array
    
    print("Detected: ", class_[0])
    
    output_indice = -1 # set it initially to -1
    
    for i in range(36):
        if(i == 0):
            max = class_[0][i]
            output_indice = 0
        else:
            if(class_[0][i] > max):
                max = class_[0][i]
                output_indice = i
                
    characters = []
    for i in range(65, 65+26):
        characters.append(chr(i))
    
    if(output_indice > 9):
        final_result = characters[(output_indice - 9) - 1]
        print("Predicted: ", final_result)
        print("value: ", max)
    else:
        print("Predicted: ", output_indice)
        print("value: ", max)
    

load_trained_model("weights.h5", "image.jpg") # Specify weights file and Test image

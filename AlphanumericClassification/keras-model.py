#import necessary modules
import keras
import tensorflow as tf
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.layers.convolutional import Conv2D


# dimensions of our images.
img_width, img_height = 32,32

train_data_dir = 'train' # train data Directory
validation_data_dir = 'test' # test data Directory


nb_train_samples = 28800  # training samples
nb_validation_samples = 7200 # test Samples 
epochs = 80      # number of epochs
batch_size = 128 

# This checks when to put channels first

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# This is the main model structure
# conv => relu => maxpool2d
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) # First convolution Layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))     #Second Convolution Layer

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  

model.add(Flatten())                     # Flatten the layers
model.add(Dense(256))                    # FC layer with 256 neurons
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(36))                    #As classes are 36, model.add(Dense(36))
model.add(Activation('softmax'))


# Compile model 
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range =0.1,
    zoom_range =0.1,
    rotation_range =10,
    width_shift_range =0.1,
    height_shift_range =0.1,
    horizontal_flip =False) # Flip = False as it we need to retain Characters

# This is the augmentation configuration we will use for testing:only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    shuffle=1,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_batch_size = batch_size
validationSamples  = 7200 
# fit the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples / batch_size, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validationSamples/validation_batch_size)

#save the weights
model.save_weights('check.h5')
model.evaluate_generator(validation_generator)





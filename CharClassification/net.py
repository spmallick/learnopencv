# import required modules
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

class Net:
    @staticmethod
    def build(width, height, depth, weightsPath=None):
        '''
        modified lenet structure
        input: input_shape (width, height, channels)
        returns: trained/loaded model
        '''
        # initialize the model
        model = Sequential()
        
        # first layer CONV => RELU => POOL
        model.add(Convolution2D(32, (3, 3), input_shape = (width, height, depth)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # second layer CONV => RELU => POOL
        model.add(Convolution2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # third layer of CONV => RELU => POOL
        model.add(Convolution2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # set of FC => RELU layers
        model.add(Flatten())

        # number of neurons in FC layer = 128
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        # as number of classes is 36
        model.add(Dense(36))
        model.add(Activation('softmax'))
        
        # if weightsPath is specified load the weights
        if weightsPath is not None:
            print('weights loaded')
            model.load_weights(weightsPath)
            # return model

        return model

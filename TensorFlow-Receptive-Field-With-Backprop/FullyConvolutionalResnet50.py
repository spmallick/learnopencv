import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.utils import data_utils

from utils import (
    BASE_WEIGHTS_PATH,
    WEIGHTS_HASHES,
    stack1,
)


# setting FC weights to the final convolutional layer
def set_conv_weights(model, feature_extractor):
    # get pre-trained ResNet50 FC weights
    dense_layer_weights = feature_extractor.layers[-1].get_weights()
    weights_list = [
        tf.reshape(
            dense_layer_weights[0], (1, 1, *dense_layer_weights[0].shape),
        ).numpy(),
        dense_layer_weights[1],
    ]
    model.get_layer(name="last_conv").set_weights(weights_list)


def fully_convolutional_resnet50(
    input_shape, num_classes=1000, pretrained_resnet=True, use_bias=True,
):
    # init input layer
    img_input = Input(shape=input_shape)

    # define basic model pipeline
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name="conv1_bn")(x)
    x = Activation("relu", name="conv1_relu")(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    # the sequence of stacked residual blocks
    x = stack1(x, 64, 3, stride1=1, name="conv2")
    x = stack1(x, 128, 4, name="conv3")
    x = stack1(x, 256, 6, name="conv4")
    x = stack1(x, 512, 3, name="conv5")

    # add avg pooling layer after feature extraction layers
    x = AveragePooling2D(pool_size=7)(x)

    # add final convolutional layer
    conv_layer_final = Conv2D(
        filters=num_classes, kernel_size=1, use_bias=use_bias, name="last_conv",
    )(x)

    # configure fully convolutional ResNet50 model
    model = training.Model(img_input, x)

    # load model weights
    if pretrained_resnet:
        model_name = "resnet50"
        # configure full file name
        file_name = model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
        # get the file hash from TF WEIGHTS_HASHES
        file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )

        model.load_weights(weights_path)

    # form final model
    model = training.Model(inputs=model.input, outputs=[conv_layer_final])

    if pretrained_resnet:
        # get model with the dense layer for further FC weights extraction
        resnet50_extractor = ResNet50(
            include_top=True, weights="imagenet", classes=num_classes,
        )
        # set ResNet50 FC-layer weights to final convolutional layer
        set_conv_weights(model=model, feature_extractor=resnet50_extractor)
    return model

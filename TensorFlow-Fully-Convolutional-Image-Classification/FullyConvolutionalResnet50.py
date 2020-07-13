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


if __name__ == "__main__":

    # read ImageNet class ids to a list of labels
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # read image
    original_image = cv2.imread("camel.jpg")

    # convert image to the RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # pre-process image
    image = preprocess_input(image)

    # convert image to NCHW tf.tensor
    image = tf.expand_dims(image, 0)

    # load modified resnet50 model with pre-trained ImageNet weights
    model = fully_convolutional_resnet50(input_shape=(image.shape[-3:]))

    # Perform inference.
    # Instead of a 1×1000 vector, we will get a
    # 1×1000×n×m output ( i.e. a probability map
    # of size n × m for each 1000 class,
    # where n and m depend on the size of the image).
    preds = model.predict(image)
    preds = tf.transpose(preds, perm=[0, 3, 1, 2])
    preds = tf.nn.softmax(preds, axis=1)
    print("Response map shape : ", preds.shape)

    # find the class with the maximum score in the n × m output map
    pred = tf.math.reduce_max(preds, axis=1)
    class_idx = tf.math.argmax(preds, axis=1)
    print(class_idx)

    row_max = tf.math.reduce_max(pred, axis=1)
    row_idx = tf.math.argmax(pred, axis=1)

    col_idx = tf.math.argmax(row_max, axis=1)

    predicted_class = tf.gather_nd(
        class_idx, (0, tf.gather_nd(row_idx, (0, col_idx[0])), col_idx[0]),
    )

    # print top predicted class
    print("Predicted Class : ", labels[predicted_class], predicted_class)

    # find the n × m score map for the predicted class
    score_map = tf.expand_dims(preds[0, predicted_class, :, :], 0).numpy()
    score_map = score_map[0]

    # resize score map to the original image size
    score_map = cv2.resize(
        score_map, (original_image.shape[1], original_image.shape[0]),
    )

    # binarize score map
    _, score_map_for_contours = cv2.threshold(
        score_map, 0.65, 1, type=cv2.THRESH_BINARY,
    )
    score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()

    # find the contour of the binary blob
    contours, _ = cv2.findContours(
        score_map_for_contours, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE,
    )

    # find bounding box around the object.
    rect = cv2.boundingRect(contours[0])

    # apply score map as a mask to original image
    score_map = score_map - np.min(score_map[:])
    score_map = score_map / np.max(score_map[:])

    score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
    masked_image = (original_image * score_map).astype(np.uint8)

    # display bounding box
    cv2.rectangle(
        masked_image, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2,
    )

    # display images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("scaled_score_map", score_map)
    cv2.imshow("activations_and_bbox", masked_image)
    cv2.waitKey(0)

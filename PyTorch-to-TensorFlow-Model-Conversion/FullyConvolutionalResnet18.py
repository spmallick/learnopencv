import cv2
import numpy as np
import tensorflow as tf
import torch
from albumentations import (
    Compose,
    Normalize,
)
from pytorch2keras.converter import pytorch_to_keras
from torch.autograd import Variable

from PyTorchFullyConvolutionalResnet18 import FullyConvolutionalResnet18


def converted_fully_convolutional_resnet18(
    input_tensor, pretrained_resnet=True,
):
    # define input tensor
    input_var = Variable(torch.FloatTensor(input_tensor))

    # get PyTorch ResNet18 model
    model_to_transfer = FullyConvolutionalResnet18(pretrained=pretrained_resnet)
    model_to_transfer.eval()

    # convert PyTorch model to Keras
    model = pytorch_to_keras(
        model_to_transfer,
        input_var,
        [input_var.shape[-3:]],
        change_ordering=True,
        verbose=False,
        name_policy="keep",
    )

    return model


if __name__ == "__main__":
    # read ImageNet class ids to a list of labels
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # read image
    original_image = cv2.imread("camel.jpg")

    # convert original image to RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # transform input image:
    transform = Compose(
        [
            Normalize(
                # subtract mean
                mean=(0.485, 0.456, 0.406),
                # divide by standard deviation
                std=(0.229, 0.224, 0.225),
            ),
        ],
    )
    # apply image transformations, (725, 1920, 3)
    image = transform(image=image)["image"]

    # NHWC: (1, 725, 1920, 3)
    predict_image = tf.expand_dims(image, 0)
    # NCHW: (1, 3, 725, 1920)
    image = np.transpose(tf.expand_dims(image, 0).numpy(), [0, 3, 1, 2])

    # get transferred torch ResNet18 with pre-trained ImageNet weights
    model = converted_fully_convolutional_resnet18(
        input_tensor=image, pretrained_resnet=True,
    )

    # Perform inference.
    # Instead of a 1×1000 vector, we will get a
    # 1×1000×n×m output ( i.e. a probability map
    # of size n × m for each 1000 class,
    # where n and m depend on the size of the image).
    preds = model.predict(predict_image)
    # NHWC: (1, 3, 8, 1000) back to NCHW: (1, 1000, 3, 8)
    preds = tf.transpose(preds, (0, 3, 1, 2))
    preds = tf.nn.softmax(preds, axis=1)
    print("Response map shape : ", preds.shape)

    # find the class with the maximum score in the n x m output map
    pred = tf.math.reduce_max(preds, axis=1)
    class_idx = tf.math.argmax(preds, axis=1)

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
        score_map, 0.25, 1, type=cv2.THRESH_BINARY,
    )

    score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()

    # Find the contour of the binary blob
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

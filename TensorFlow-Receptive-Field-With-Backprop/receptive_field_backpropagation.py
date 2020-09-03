from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
)

from FullyConvolutionalResnet50 import fully_convolutional_resnet50

Rect = namedtuple("Rect", "x1 y1 x2 y2")


def backprop_receptive_field(
    image, predicted_class, scoremap, use_max_activation=False,
):
    model = fully_convolutional_resnet50(
        input_shape=(image.shape[-3:]), pretrained_resnet=False,
    )

    for module in model.layers:
        try:
            if isinstance(module, Conv2D):
                conv_weights = np.full(module.get_weights()[0].shape, 0.005)
                if len(module.get_weights()) > 1:
                    conv_biases = np.full(module.get_weights()[1].shape, 0.0)
                    module.set_weights([conv_weights, conv_biases])
                # cases when use_bias = False
                else:
                    module.set_weights([conv_weights])
            if isinstance(module, BatchNormalization):
                # weight sequence: gamma, beta, running mean, running variance
                bn_weights = [
                    module.get_weights()[0],
                    module.get_weights()[1],
                    np.full(module.get_weights()[2].shape, 0.0),
                    np.full(module.get_weights()[3].shape, 1.0),
                ]
                module.set_weights(bn_weights)
        except:
            pass

    input = tf.ones_like(image)
    out = model.predict(image)
    receptive_field_mask = tf.Variable(tf.zeros_like(out))

    if not use_max_activation:
        receptive_field_mask[:, :, :, predicted_class].assign(scoremap)
    else:
        scoremap_max_row_values = tf.math.reduce_max(scoremap, axis=1)
        max_row_id = tf.math.argmax(scoremap, axis=1)
        max_col_id = tf.math.argmax(scoremap_max_row_values, axis=1).numpy()[0]
        max_row_id = max_row_id[0, max_col_id].numpy()
        print(
            "Coords of the max activation:", max_row_id, max_col_id,
        )
        # update gradient
        receptive_field_mask = tf.tensor_scatter_nd_update(
            receptive_field_mask, [(0, max_row_id, max_col_id, 0)], [1],
        )

    grads = []
    with tf.GradientTape() as tf_gradient_tape:
        tf_gradient_tape.watch(input)
        # get the predictions
        preds = model(input)
        # apply the mask
        pseudo_loss = preds * receptive_field_mask
        pseudo_loss = K.mean(pseudo_loss)
        # get gradient
        grad = tf_gradient_tape.gradient(pseudo_loss, input)
        grad = tf.transpose(grad, perm=[0, 3, 1, 2])
        grads.append(grad)
    return grads[0][0, 0]


def find_rect(activations):
    # Dilate and erode the activations to remove grid-like artifacts
    kernel = np.ones((5, 5), np.uint8)
    activations = cv2.dilate(activations, kernel=kernel)
    activations = cv2.erode(activations, kernel=kernel)

    # Binarize the activations
    _, activations = cv2.threshold(activations, 0.65, 1, type=cv2.THRESH_BINARY)
    activations = activations.astype(np.uint8).copy()

    # Find the contour of the binary blob
    contours, _ = cv2.findContours(
        activations, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE,
    )

    # Find bounding box around the object.
    rect = cv2.boundingRect(contours[0])
    return Rect(rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])


def normalize(activations):
    activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations


def visualize_activations(image, activations, show_bounding_rect=False):
    activations = normalize(activations)

    activations_multichannel = np.stack([activations, activations, activations], axis=2)
    masked_image = (image * activations_multichannel).astype(np.uint8)

    if show_bounding_rect:
        rect = find_rect(activations.numpy())
        cv2.rectangle(
            masked_image,
            (rect.x1, rect.y1),
            (rect.x2, rect.y2),
            color=(0, 0, 255),
            thickness=2,
        )

    return masked_image


def run_resnet_inference(original_image):

    # read ImageNet class ids to a list of labels
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # convert image to the RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # pre-process image
    image = preprocess_input(image)

    # convert image to NCHW tf.tensor
    image = tf.expand_dims(image, 0)

    # load resnet50 model with pretrained ImageNet weights
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

    # find class with the maximum score in the n × m output map
    pred = tf.math.reduce_max(preds, axis=1)
    class_idx = tf.math.argmax(preds, axis=1)

    row_max = tf.math.reduce_max(pred, axis=1)
    row_idx = tf.math.argmax(pred, axis=1)

    col_idx = tf.math.argmax(row_max, axis=1)

    predicted_class = tf.gather_nd(
        class_idx, (0, tf.gather_nd(row_idx, (0, col_idx[0])), col_idx[0]),
    )

    # print the top predicted class
    print("Predicted Class : ", labels[predicted_class], predicted_class)

    # find the n × m score map for the predicted class
    score_map = tf.expand_dims(preds[0, predicted_class, :, :], 0).numpy()
    print("Score Map shape : ", score_map.shape)

    # compute the receptive filed for max activation pixel
    receptive_field_max_activation = backprop_receptive_field(
        image,
        scoremap=score_map,
        predicted_class=predicted_class,
        use_max_activation=True,
    )
    # compute the receptive filed for the whole image
    receptive_field_image = backprop_receptive_field(
        image,
        scoremap=score_map,
        predicted_class=predicted_class,
        use_max_activation=False,
    )

    # resize score map to the original image size
    score_map = score_map[0]
    score_map = cv2.resize(
        score_map, (original_image.shape[1], original_image.shape[0]),
    )

    # display the images
    cv2.imshow("Original Image", original_image)
    cv2.imshow(
        "Score map: activations and bbox",
        visualize_activations(original_image, score_map),
    )
    cv2.imshow(
        "receptive_field_max_activation",
        visualize_activations(
            original_image, receptive_field_max_activation, show_bounding_rect=True,
        ),
    )
    cv2.imshow(
        "receptive_field_the_whole_image",
        visualize_activations(
            original_image, receptive_field_image, show_bounding_rect=True,
        ),
    )
    cv2.waitKey(0)


def main():
    # read the image
    image_path = "camel.jpg"
    image = cv2.imread(image_path)

    # run inference
    run_resnet_inference(image)


if __name__ == "__main__":
    main()

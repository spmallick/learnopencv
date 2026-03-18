import os
import tempfile

# onnx2keras still relies on legacy Keras layer APIs, so enable tf-keras
# before TensorFlow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import onnx
import tensorflow as tf
import torch
from albumentations import Compose, Normalize
from onnx2keras import onnx_to_keras

from PyTorchFullyConvolutionalResnet18 import FullyConvolutionalResnet18


def converted_fully_convolutional_resnet18(
    input_tensor, pretrained_resnet=True,
):
    # Build the fully-convolutional PyTorch ResNet18 and freeze it for export.
    model_to_transfer = FullyConvolutionalResnet18(pretrained=pretrained_resnet)
    model_to_transfer.eval()

    input_var = torch.as_tensor(input_tensor, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        onnx_path = handle.name

    try:
        # pytorch2keras no longer works with current ONNX releases, so convert
        # through an intermediate ONNX file instead.
        torch.onnx.export(
            model_to_transfer,
            input_var,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        onnx_model = onnx.load(onnx_path)
        return onnx_to_keras(
            onnx_model,
            ["input"],
            change_ordering=True,
            verbose=False,
            # The exported ONNX graph contains names that begin with "/".
            # Renumbering keeps generated Keras layer names valid.
            name_policy="renumerate",
        )
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)


if __name__ == "__main__":
    # Read ImageNet class ids to a list of labels.
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # Read image and convert the OpenCV BGR array to RGB.
    original_image = cv2.imread("camel.jpg")
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Normalize the input exactly like the pretrained ResNet18 expects.
    transform = Compose(
        [
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ],
    )
    image = transform(image=image)["image"]

    # TensorFlow inference uses NHWC while the exported PyTorch model expects
    # NCHW, so keep both layouts.
    predict_image = tf.expand_dims(image, 0)
    image = np.transpose(tf.expand_dims(image, 0).numpy(), [0, 3, 1, 2])

    # Get the converted TensorFlow/Keras model from the PyTorch checkpoint.
    model = converted_fully_convolutional_resnet18(
        input_tensor=image, pretrained_resnet=True,
    )

    # Instead of a 1 x 1000 vector, the fully-convolutional model returns a
    # class score map for each spatial location.
    preds = model.predict(predict_image)
    preds = tf.transpose(preds, (0, 3, 1, 2))
    preds = tf.nn.softmax(preds, axis=1)
    print("Response map shape : ", preds.shape)

    # Find the class and spatial position with the maximum response.
    pred = tf.math.reduce_max(preds, axis=1)
    class_idx = tf.math.argmax(preds, axis=1)

    row_max = tf.math.reduce_max(pred, axis=1)
    row_idx = tf.math.argmax(pred, axis=1)

    col_idx = tf.math.argmax(row_max, axis=1)

    predicted_class = tf.gather_nd(
        class_idx, (0, tf.gather_nd(row_idx, (0, col_idx[0])), col_idx[0]),
    )

    # Print the top predicted class.
    print("Predicted Class : ", labels[predicted_class], predicted_class)

    # Extract the score map for the predicted class and resize it to the
    # original image resolution.
    score_map = tf.expand_dims(preds[0, predicted_class, :, :], 0).numpy()
    score_map = score_map[0]
    score_map = cv2.resize(
        score_map, (original_image.shape[1], original_image.shape[0]),
    )

    # Binarize the score map and recover a bounding box from the largest blob.
    _, score_map_for_contours = cv2.threshold(
        score_map, 0.25, 1, type=cv2.THRESH_BINARY,
    )

    score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()
    contours, _ = cv2.findContours(
        score_map_for_contours, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE,
    )
    rect = cv2.boundingRect(contours[0])

    # Use the normalized score map as a soft mask for visualization.
    score_map = score_map - np.min(score_map[:])
    score_map = score_map / np.max(score_map[:])

    score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
    masked_image = (original_image * score_map).astype(np.uint8)

    cv2.rectangle(
        masked_image, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2,
    )

    # Display the original image, score map and localized activation result.
    cv2.imshow("Original Image", original_image)
    cv2.imshow("scaled_score_map", score_map)
    cv2.imshow("activations_and_bbox", masked_image)
    cv2.waitKey(0)

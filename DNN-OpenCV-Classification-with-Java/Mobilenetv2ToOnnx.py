import argparse
import os

import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from albumentations import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
)
from torchvision import models


def compare_pytorch_onnx(
    original_model_preds, onnx_model_path, input_image,
):
    # get onnx result
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    onnx_result = session.run([], {input_name: input_image})
    onnx_result = np.squeeze(onnx_result, axis=0)

    print("Checking PyTorch model and converted ONNX model outputs ... ")
    for test_onnx_result, gold_result in zip(onnx_result, original_model_preds):
        np.testing.assert_almost_equal(
            gold_result, test_onnx_result, decimal=3,
        )
    print("PyTorch and ONNX output values are equal! \n")


def get_onnx_model(
    original_model, input_image, model_path="models", model_name="pytorch_mobilenet",
):
    # create model root dir
    os.makedirs(model_path, exist_ok=True)

    model_name = os.path.join(model_path, model_name + ".onnx")

    torch.onnx.export(
        original_model, torch.Tensor(input_image), model_name, verbose=True,
    )
    print("ONNX model was successfully generated: {} \n".format(model_name))

    return model_name


def get_preprocessed_image(image_name):
    # read image
    original_image = cv2.imread(image_name)

    # convert original image to RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # transform input image:
    # 1. resize the image
    # 2. crop the image
    # 3. normalize: subtract mean and divide by standard deviation
    transform = Compose(
        [
            Resize(height=256, width=256),
            CenterCrop(224, 224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    image = transform(image=image)["image"]

    # change the order of channels
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image, axis=0)


def get_predicted_class(pytorch_preds):
    # read ImageNet class id to name mapping
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    # find the class with the maximum score
    pytorch_class_idx = np.argmax(pytorch_preds, axis=1)
    predicted_pytorch_label = labels[pytorch_class_idx[0]]

    # print top predicted class
    print("Predicted class by PyTorch model: ", predicted_pytorch_label)


def get_execution_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_image",
        type=str,
        help="Define the full input image path, including its name",
        default="images/coffee.jpg",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # get the test case parameters
    args = get_execution_arguments()

    # read and process the input image
    image = get_preprocessed_image(image_name=args.input_image)

    # obtain original model
    pytorch_model = models.mobilenet_v2(pretrained=True)

    # provide inference of the original PyTorch model
    pytorch_model.eval()
    pytorch_predictions = pytorch_model(torch.Tensor(image)).detach().numpy()

    # obtain OpenCV generated ONNX model
    onnx_model_path = get_onnx_model(original_model=pytorch_model, input_image=image)

    # check if conversion succeeded
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # check onnx model output
    compare_pytorch_onnx(
        pytorch_predictions, onnx_model_path, image,
    )

    # decode classification results
    get_predicted_class(pytorch_preds=pytorch_predictions)

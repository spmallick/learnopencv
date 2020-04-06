import numpy as np
import cv2
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import transforms
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18


Rect = namedtuple('Rect', 'x1 y1 x2 y2')


def backprop_receptive_field(image, predicted_class, scoremap, use_max_activation=True):
    model = FullyConvolutionalResnet18()
    model = model.train()
    for module in model.modules():
        try:
            nn.init.constant_(module.weight, 0.05) # inference overflows with ones
            nn.init.zeros_(module.bias)
            nn.init.zeros_(module.running_mean)
            nn.init.ones_(module.running_var)
        except:
            pass

        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

    input = torch.ones_like(image, requires_grad=True)

    out = model(input)
    grad = torch.zeros_like(out, requires_grad=True)

    if not use_max_activation:
        grad[0, predicted_class] = scoremap
    else:
        scoremap_max_row_values, max_row_id = torch.max(scoremap, dim=1)
        _, max_col_id = torch.max(scoremap_max_row_values, dim=1)
        max_row_id = max_row_id[0, max_col_id]
        print('Coords of the max activation:', max_row_id.item(), max_col_id.item())

        grad[0, 0, max_row_id, max_col_id] = 1

    out.backward(gradient=grad)
    gradient_of_input = input.grad[0, 0].data.numpy()
    gradient_of_input = gradient_of_input / np.amax(gradient_of_input)

    return gradient_of_input


def find_rect(activations):
    # Dilate and erode the activations to remove grid-like artifacts
    kernel = np.ones((5, 5), np.uint8)
    activations = cv2.dilate(activations, kernel=kernel)
    activations = cv2.erode(activations, kernel=kernel)

    # Binarize the activations
    _, activations = cv2.threshold(activations, 0.25, 1, type=cv2.THRESH_BINARY)
    activations = activations.astype(np.uint8).copy()

    # Find the countour of the binary blob
    contours, _ = cv2.findContours(activations, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

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
        rect = find_rect(activations)
        cv2.rectangle(masked_image, (rect.x1, rect.y1), (rect.x2, rect.y2), color=(0, 0, 255), thickness=2)

    return masked_image


def run_resnet_inference(original_image):
    # Read ImageNet class id to name mapping
    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    # Convert original image to RGB format
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Transform input image
    # 1. Convert to Tensor
    # 2. Subtract mean
    # 3. Divide by standard deviation

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Subtract mean
            std=[0.229, 0.224, 0.225]  # Divide by standard deviation
        )])

    image = transform(image)
    image = image.unsqueeze(0)

    # Load modified resnet18 model with pretrained ImageNet weights
    model = FullyConvolutionalResnet18(pretrained=True).eval()

    with torch.no_grad():
        # Perform the inference.
        # Instead of a 1x1000 vector, we will get a
        # 1x1000xnxm output ( i.e. a probabibility map
        # of size n x m for each 1000 class,
        # where n and m depend on the size of the image.)
        preds = model(image)
        preds = torch.softmax(preds, dim=1)

        print('Response map shape : ', preds.shape)

        # Find the class with the maximum score in the n x m output map
        pred, class_idx = torch.max(preds, dim=1)

        row_max, row_idx = torch.max(pred, dim=1)
        col_max, col_idx = torch.max(row_max, dim=1)
        predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]

        # Print the top predicted class
        print('Predicted Class : ', labels[predicted_class], predicted_class)

        # Find the n x m score map for the predicted class
        score_map = preds[0, predicted_class, :, :].cpu()
        print('Score Map shape : ', score_map.shape)

    # Compute the receptive filed for the inference result
    receptive_field_map = backprop_receptive_field(image, scoremap=score_map, predicted_class=predicted_class)

    # Resize score map to the original image size
    score_map = score_map.numpy()[0]
    score_map = cv2.resize(score_map, (original_image.shape[1], original_image.shape[0]))

    # Display the images
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Score map: activations and bbox", visualize_activations(original_image, score_map))
    cv2.imshow("receptive_field_max_activation", visualize_activations(original_image, receptive_field_map, show_bounding_rect=True))
    cv2.waitKey(0)


def main():
    # Read the image
    image_path = 'camel.jpg'
    image = cv2.imread(image_path)

    run_resnet_inference(image)


if __name__ == "__main__":
    main()

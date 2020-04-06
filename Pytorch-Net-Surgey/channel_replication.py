import torch

from torchvision import models

import cv2
import numpy as np


def classify_image(image):
    model = models.resnet18(pretrained=True).eval()

    imagenet_means = [0.485, 0.456, 0.406][::-1]
    imagenet_stds = [0.229, 0.224, 0.225][::-1]

    image = (image / 255.0 - imagenet_means) / imagenet_stds
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        pred, class_idx = torch.max(preds, dim=1)
        print('Class id: {}, confidence: {}'.format(class_idx.item(), pred.item()))

def classify_grayscale():
    image = cv2.imread("dog-basset-hound.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = np.stack((image, image, image), axis=2)
    classify_image(image)

def classify_colorful():
    image = cv2.imread("dog-basset-hound.jpg", cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))
    classify_image(image)


if __name__ == "__main__":
    classify_colorful()
    classify_grayscale()

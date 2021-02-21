import torch

from torchvision import models

import cv2
import numpy as np


def classify_image(image, model):
    model = model.eval()
    image = torch.tensor(image, dtype=torch.float32)

    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        pred, class_idx = torch.max(preds, dim=1)
        print('Class id: {}, confidence: {}'.format(class_idx.item(), pred.item()))


def classify_grayscale():
    image = cv2.imread("dog-basset-hound.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    imagenet_means = torch.tensor([0.485, 0.456, 0.406][::-1])
    imagenet_stds = torch.tensor([0.229, 0.224, 0.225][::-1])
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, bias=False),
        torch.nn.BatchNorm2d(num_features=3),
        models.resnet18(pretrained=True),
    )
    model[0].weight.data.fill_(1.0)
    model[1].running_mean.copy_(imagenet_means)
    model[1].running_var.copy_(imagenet_stds)

    classify_image(image, model)


def classify_colorful():
    image = cv2.imread("dog-basset-hound.jpg", cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))

    imagenet_means = [0.485, 0.456, 0.406][::-1]
    imagenet_stds = [0.229, 0.224, 0.225][::-1]

    image = (image / 255.0 - imagenet_means) / imagenet_stds
    image = image.transpose(2, 0, 1)
    model = models.resnet18(pretrained=True)
    classify_image(image, model)


if __name__ == "__main__":
    classify_colorful()
    classify_grayscale()

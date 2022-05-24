import torch
from torch import nn

from torchvision import models
from torch.hub import load_state_dict_from_url

import time
from tqdm import tqdm

import cv2
import numpy as np


class Model(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):

        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7))
        self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)

        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.last_conv(x)
        return x


def surgery():
    original_image = cv2.imread("dog-basset-hound.jpg", cv2.IMREAD_UNCHANGED)
    original_image = cv2.resize(original_image, None, None, fx=1 / 2.0, fy=1 / 2.0)
    cv2.imshow("original", original_image)
    image = original_image.copy()
    model = Model(pretrained=True).eval()

    imagenet_means = [0.485, 0.456, 0.406][::-1]
    imagenet_stds = [0.229, 0.224, 0.225][::-1]

    image = (image / 255.0 - imagenet_means) / imagenet_stds
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

    image = image.unsqueeze(0)
    with torch.no_grad():
        preds = model(image)
        pred, class_idx = torch.max(preds, dim=1)
        row_max, row_idx = torch.max(pred, dim=1)
        col_max, col_idx = torch.max(row_max, dim=1)
        print('Most confident class: ', class_idx[0, col_idx, row_idx[0, col_idx]].item())

        preds = torch.softmax(preds, dim=1)
        score_map = preds[0, class_idx[0, col_idx, row_idx[0, col_idx]], :, :].cpu().numpy()
        score_map = score_map[0]
        score_map = np.expand_dims(score_map, -1)
        score_map = np.repeat(score_map, 3, axis=2)
        score_map = cv2.resize(score_map, (original_image.shape[1], original_image.shape[0]))
        cv2.imshow("activations", (original_image * score_map).astype(np.uint8))
        cv2.waitKey(0)
        cv2.imwrite("activations_with_pooling.png", (original_image * score_map).astype(np.uint8))


if __name__ == "__main__":
    surgery()

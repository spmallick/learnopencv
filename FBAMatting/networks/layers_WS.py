import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        # std = (weight).view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        std = (
            torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(
                -1, 1, 1, 1,
            )
            + 1e-5
        )
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
        )


def BatchNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=32)

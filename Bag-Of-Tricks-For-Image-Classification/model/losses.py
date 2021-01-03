# MIT License
# Copyright (c) 2018 Haitong Li


import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha, T, criterion):
        super().__init__()
        self.criterion = criterion
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")
        self.alpha = alpha
        self.T = T

    def forward(self, input, target, teacher_target):
        loss = self.KLDivLoss(
            F.log_softmax(input / self.T, dim=1),
            F.softmax(teacher_target / self.T, dim=1),
        ) * (self.alpha * self.T * self.T) + self.criterion(input, target) * (
            1.0 - self.alpha
        )
        return loss


class MixUpAugmentationLoss(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, input, target, *args):
        # Validation step
        if isinstance(target, torch.Tensor):
            return self.criterion(input, target, *args)
        target_a, target_b, lmbd = target
        return lmbd * self.criterion(input, target_a, *args) + (
            1 - lmbd
        ) * self.criterion(input, target_b, *args)


# Based on https://github.com/pytorch/pytorch/issues/7455
class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = n_classes
        self.dim = dim

    def forward(self, output, target, *args):
        output = output.log_softmax(dim=self.dim)
        with torch.no_grad():
            # Create matrix with shapes batch_size x n_classes
            true_dist = torch.zeros_like(output)
            # Initialize all elements with epsilon / N - 1
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # Fill correct class for each sample in the batch with 1 - epsilon
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * output, dim=self.dim))

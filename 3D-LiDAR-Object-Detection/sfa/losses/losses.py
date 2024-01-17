# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# Modified by Nguyen Mau Dung (2020.08.09)
# ------------------------------------------------------------------------------

import os
import sys
import math

import torch.nn as nn
import torch
import torch.nn.functional as F

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.torch_utils import to_cpu, _sigmoid


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _neg_loss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss_Balanced(nn.Module):
    """Balanced L1 Loss
    paper: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    Code refer from: https://github.com/OceanPang/Libra_R-CNN
    """

    def __init__(self, alpha=0.5, gamma=1.5, beta=1.0):
        super(L1Loss_Balanced, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        assert beta > 0
        self.beta = beta

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = self.balanced_l1_loss(pred * mask, target * mask)
        loss = loss.sum() / (mask.sum() + 1e-4)

        return loss

    def balanced_l1_loss(self, pred, target):
        assert pred.size() == target.size() and target.numel() > 0

        diff = torch.abs(pred - target)
        b = math.exp(self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta,
                           self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff,
                           self.gamma * diff + self.gamma / b - self.alpha * self.beta)

        return loss


class Compute_Loss(nn.Module):
    def __init__(self, device):
        super(Compute_Loss, self).__init__()
        self.device = device
        self.focal_loss = FocalLoss()
        self.l1_loss = L1Loss()
        self.l1_loss_balanced = L1Loss_Balanced(alpha=0.5, gamma=1.5, beta=1.0)
        self.weight_hm_cen = 1.
        self.weight_z_coor, self.weight_cenoff, self.weight_dim, self.weight_direction = 1., 1., 1., 1.

    def forward(self, outputs, tg):
        # tg: targets
        outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
        outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])

        l_hm_cen = self.focal_loss(outputs['hm_cen'], tg['hm_cen'])
        l_cen_offset = self.l1_loss(outputs['cen_offset'], tg['obj_mask'], tg['indices_center'], tg['cen_offset'])
        l_direction = self.l1_loss(outputs['direction'], tg['obj_mask'], tg['indices_center'], tg['direction'])
        # Apply the L1_loss balanced for z coor and dimension regression
        l_z_coor = self.l1_loss_balanced(outputs['z_coor'], tg['obj_mask'], tg['indices_center'], tg['z_coor'])
        l_dim = self.l1_loss_balanced(outputs['dim'], tg['obj_mask'], tg['indices_center'], tg['dim'])

        total_loss = l_hm_cen * self.weight_hm_cen + l_cen_offset * self.weight_cenoff + \
                     l_dim * self.weight_dim + l_direction * self.weight_direction + \
                     l_z_coor * self.weight_z_coor

        loss_stats = {
            'total_loss': to_cpu(total_loss).item(),
            'hm_cen_loss': to_cpu(l_hm_cen).item(),
            'cen_offset_loss': to_cpu(l_cen_offset).item(),
            'dim_loss': to_cpu(l_dim).item(),
            'direction_loss': to_cpu(l_direction).item(),
            'z_coor_loss': to_cpu(l_z_coor).item(),
        }

        return total_loss, loss_stats

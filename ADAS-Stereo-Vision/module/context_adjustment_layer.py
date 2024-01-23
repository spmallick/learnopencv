#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm


class ContextAdjustmentLayer(nn.Module):
    """
    Adjust the disp and occ based on image context, design loosely follows https://github.com/JiahuiYu/wdsr_ntire2018
    """

    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super().__init__()
        self.num_blocks = num_blocks

        # disp head
        self.in_conv = nn.Conv2d(4, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

        # occ head
        self.occ_head = nn.Sequential(
            weight_norm(nn.Conv2d(1 + 3, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, disp_raw: Tensor, occ_raw: Tensor, img: Tensor):
        """
        :param disp_raw: raw disparity, [N,1,H,W]
        :param occ_raw: raw occlusion mask, [N,1,H,W]
        :param img: input left image, [N,3,H,W]
        :return:
            disp_final: final disparity [N,1,H,W]
            occ_final: final occlusion [N,1,H,W] 
        """""
        feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
        for layer in self.layers:
            feat = layer(feat, disp_raw)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res

        occ_final = self.occ_head(torch.cat([occ_raw, img], dim=1))

        return disp_final, occ_final


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale


def build_context_adjustment_layer(args):
    if args.context_adjustment_layer == 'cal':
        return ContextAdjustmentLayer(args.cal_num_blocks, args.cal_feat_dim,
                                      args.cal_expansion_ratio)
    elif args.context_adjustment_layer == 'none':
        return None
    else:
        raise ValueError(f'Context adjustment layer option not recognized: {args.context_adjustment_layer}')

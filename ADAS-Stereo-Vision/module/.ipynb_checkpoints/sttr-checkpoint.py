#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch.nn as nn

from module.feat_extractor_backbone import build_backbone
from module.feat_extractor_tokenizer import build_tokenizer
from module.pos_encoder import build_position_encoding
from module.regression_head import build_regression_head
from module.transformer import build_transformer
from utilities.misc import batched_index_select, NestedTensor


class STTR(nn.Module):
    """
    STTR: it consists of
        - backbone: contracting path of feature descriptor
        - tokenizer: expanding path of feature descriptor
        - pos_encoder: generates relative sine pos encoding
        - transformer: computes self and cross attention
        - regression_head: regresses disparity and occlusion, including optimal transport
    """

    def __init__(self, args):
        super(STTR, self).__init__()
        layer_channel = [64, 128, 128]

        self.backbone = build_backbone(args)
        self.tokenizer = build_tokenizer(args, layer_channel)
        self.pos_encoder = build_position_encoding(args)
        self.transformer = build_transformer(args)
        self.regression_head = build_regression_head(args)

        self._reset_parameters()
        self._disable_batchnorm_tracking()
        self._relu_inplace()

    def _reset_parameters(self):
        """
        xavier initialize all params
        """
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def _disable_batchnorm_tracking(self):
        """
        disable Batchnorm tracking stats to reduce dependency on dataset (this acts as InstanceNorm with affine when batch size is 1)
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                # fix the bug mentioned in https://github.com/mli0603/stereo-transformer/issues/8
                m.running_mean = None
                m.running_var = None

    def _relu_inplace(self):
        """
        make all ReLU inplace
        """
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x: NestedTensor):
        """
        :param x: input data
        :return:
            a dictionary object with keys
            - "disp_pred" [N,H,W]: predicted disparity
            - "occ_pred" [N,H,W]: predicted occlusion mask
            - "disp_pred_low_res" [N,H//s,W//s]: predicted low res (raw) disparity
        """
        bs, _, h, w = x.left.size()

        # extract features
        feat = self.backbone(x)  # concatenate left and right along the dim=0
        tokens = self.tokenizer(feat)  # 2NxCxHxW
        pos_enc = self.pos_encoder(x)  # NxCxHx2W-1

        # separate left and right
        feat_left = tokens[:bs]
        feat_right = tokens[bs:]  # NxCxHxW

        # downsample
        if x.sampled_cols is not None:
            feat_left = batched_index_select(feat_left, 3, x.sampled_cols)
            feat_right = batched_index_select(feat_right, 3, x.sampled_cols)
        if x.sampled_rows is not None:
            feat_left = batched_index_select(feat_left, 2, x.sampled_rows)
            feat_right = batched_index_select(feat_right, 2, x.sampled_rows)

        # transformer
        attn_weight = self.transformer(feat_left, feat_right, pos_enc)

        # regress disparity and occlusion
        output = self.regression_head(attn_weight, x)

        return output

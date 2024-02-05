#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from module.context_adjustment_layer import build_context_adjustment_layer
from utilities.misc import batched_index_select, torch_1d_sample, NestedTensor


class RegressionHead(nn.Module):
    """
    Regress disparity and occlusion mask
    """

    def __init__(self, cal: nn.Module, ot: bool = True):
        super(RegressionHead, self).__init__()
        self.cal = cal
        self.ot = ot
        self.phi = nn.Parameter(torch.tensor(0.0, requires_grad=True))  # dustbin cost

    def _compute_unscaled_pos_shift(self, w: int, device: torch.device):
        """
        Compute relative difference between each pixel location from left image to right image, to be used to calculate
        disparity

        :param w: image width
        :param device: torch device
        :return: relative pos shifts
        """
        pos_r = torch.linspace(0, w - 1, w)[None, None, None, :].to(device)  # 1 x 1 x 1 x W_right
        pos_l = torch.linspace(0, w - 1, w)[None, None, :, None].to(device)  # 1 x 1 x W_left x1
        pos = pos_l - pos_r
        pos[pos < 0] = 0
        return pos

    def _compute_low_res_disp(self, pos_shift: Tensor, attn_weight: Tensor, occ_mask: Tensor):
        """
        Compute low res disparity using the attention weight by finding the most attended pixel and regress within the 3px window

        :param pos_shift: relative pos shift (computed from _compute_unscaled_pos_shift), [1,1,W,W]
        :param attn_weight: attention (computed from _optimal_transport), [N,H,W,W]
        :param occ_mask: ground truth occlusion mask, [N,H,W]
        :return: low res disparity, [N,H,W] and attended similarity sum, [N,H,W]
        """

        # find high response area
        high_response = torch.argmax(attn_weight, dim=-1)  # NxHxW

        # build 3 px local window
        response_range = torch.stack([high_response - 1, high_response, high_response + 1], dim=-1)  # NxHxWx3

        # attention with re-weighting
        attn_weight_pad = F.pad(attn_weight, [1, 1], value=0.0)  # N x Hx W_left x (W_right+2)
        attn_weight_rw = torch.gather(attn_weight_pad, -1, response_range + 1)  # offset range by 1, N x H x W_left x 3

        # compute sum of attention
        norm = attn_weight_rw.sum(-1, keepdim=True)
        if occ_mask is None:
            norm[norm < 0.1] = 1.0
        else:
            norm[occ_mask, :] = 1.0  # set occluded region norm to be 1.0 to avoid division by 0

        # re-normalize to 1
        attn_weight_rw = attn_weight_rw / norm  # re-sum to 1
        pos_pad = F.pad(pos_shift, [1, 1]).expand_as(attn_weight_pad)
        pos_rw = torch.gather(pos_pad, -1, response_range + 1)

        # compute low res disparity
        disp_pred_low_res = (attn_weight_rw * pos_rw)  # NxHxW

        return disp_pred_low_res.sum(-1), norm

    def _compute_gt_location(self, scale: int, sampled_cols: Tensor, sampled_rows: Tensor,
                             attn_weight: Tensor, disp: Tensor):
        """
        Find target locations using ground truth disparity.
        Find ground truth response at those locations using attention weight.

        :param scale: high-res to low-res disparity scale
        :param sampled_cols: index to downsample columns
        :param sampled_rows: index to downsample rows
        :param attn_weight: attention weight (output from _optimal_transport), [N,H,W,W]
        :param disp: ground truth disparity
        :return: response at ground truth location [N,H,W,1] and target ground truth locations [N,H,W,1]
        """
        # compute target location at full res
        _, _, w = disp.size()
        pos_l = torch.linspace(0, w - 1, w)[None,].to(disp.device)  # 1 x 1 x W (left)
        target = (pos_l - disp)[..., None]  # N x H x W (left) x 1

        if sampled_cols is not None:
            target = batched_index_select(target, 2, sampled_cols)
        if sampled_rows is not None:
            target = batched_index_select(target, 1, sampled_rows)
        target = target / scale  # scale target location

        # compute ground truth response location for rr loss
        gt_response = torch_1d_sample(attn_weight, target, 'linear')  # NxHxW_left

        return gt_response, target

    def _upsample(self, x: NestedTensor, disp_pred: Tensor, occ_pred: Tensor, scale: int):
        """
        Upsample the raw prediction to full resolution

        :param x: input data
        :param disp_pred: predicted disp at low res
        :param occ_pred: predicted occlusion at low res
        :param scale: high-res to low-res disparity scale
        :return: high res disp and occ prediction
        """
        _, _, h, w = x.left.size()

        # scale disparity
        disp_pred_attn = disp_pred * scale

        # upsample
        disp_pred = F.interpolate(disp_pred_attn[None,], size=(h, w), mode='nearest')  # N x 1 x H x W
        occ_pred = F.interpolate(occ_pred[None,], size=(h, w), mode='nearest')  # N x 1 x H x W

        if self.cal is not None:
            # normalize disparity
            eps = 1e-6
            mean_disp_pred = disp_pred.mean()
            std_disp_pred = disp_pred.std() + eps
            disp_pred_normalized = (disp_pred - mean_disp_pred) / std_disp_pred

            # normalize occlusion mask
            occ_pred_normalized = (occ_pred - 0.5) / 0.5

            disp_pred_normalized, occ_pred = self.cal(disp_pred_normalized, occ_pred_normalized, x.left)  # N x H x W

            disp_pred_final = disp_pred_normalized * std_disp_pred + mean_disp_pred
        else:
            disp_pred_final = disp_pred.squeeze(1)
            disp_pred_attn = disp_pred_attn.squeeze(1)

        return disp_pred_final.squeeze(1), disp_pred_attn.squeeze(1), occ_pred.squeeze(1)

    def _sinkhorn(self, attn: Tensor, log_mu: Tensor, log_nu: Tensor, iters: int):
        """
        Sinkhorn Normalization in Log-space as matrix scaling problem.
        Regularization strength is set to 1 to avoid manual checking for numerical issues
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)

        :param attn: input attention weight, [N,H,W+1,W+1]
        :param log_mu: marginal distribution of left image, [N,H,W+1]
        :param log_nu: marginal distribution of right image, [N,H,W+1]
        :param iters: number of iterations
        :return: updated attention weight
        """

        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for idx in range(iters):
            # scale v first then u to ensure row sum is 1, col sum slightly larger than 1
            v = log_nu - torch.logsumexp(attn + u.unsqueeze(3), dim=2)
            u = log_mu - torch.logsumexp(attn + v.unsqueeze(2), dim=3)

        return attn + u.unsqueeze(3) + v.unsqueeze(2)

    def _optimal_transport(self, attn: Tensor, iters: int):
        """
        Perform Differentiable Optimal Transport in Log-space for stability
        Adapted from SuperGlue (https://github.com/magicleap/SuperGluePretrainedNetwork)

        :param attn: raw attention weight, [N,H,W,W]
        :param iters: number of iterations to run sinkhorn
        :return: updated attention weight, [N,H,W+1,W+1]
        """
        bs, h, w, _ = attn.shape

        # set marginal to be uniform distribution
        marginal = torch.cat([torch.ones([w]), torch.tensor([w]).float()]) / (2 * w)
        log_mu = marginal.log().to(attn.device).expand(bs, h, w + 1)
        log_nu = marginal.log().to(attn.device).expand(bs, h, w + 1)

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        # sinkhorn
        attn_ot = self._sinkhorn(similarity_matrix, log_mu, log_nu, iters)

        # convert back from log space, recover probabilities by normalization 2W
        attn_ot = (attn_ot + torch.log(torch.tensor([2.0 * w]).to(attn.device))).exp()

        return attn_ot

    def _softmax(self, attn: Tensor):
        """
        Alternative to optimal transport

        :param attn: raw attention weight, [N,H,W,W]
        :return: updated attention weight, [N,H,W+1,W+1]
        """
        bs, h, w, _ = attn.shape

        # add dustbins
        similarity_matrix = torch.cat([attn, self.phi.expand(bs, h, w, 1).to(attn.device)], -1)
        similarity_matrix = torch.cat([similarity_matrix, self.phi.expand(bs, h, 1, w + 1).to(attn.device)], -2)

        attn_softmax = F.softmax(similarity_matrix, dim=-1)

        return attn_softmax

    def _compute_low_res_occ(self, matched_attn: Tensor):
        """
        Compute low res occlusion by using inverse of the matched values

        :param matched_attn: updated attention weight without dustbins, [N,H,W,W]
        :return: low res occlusion map, [N,H,W]
        """
        occ_pred = 1.0 - matched_attn
        return occ_pred.squeeze(-1)

    def forward(self, attn_weight: Tensor, x: NestedTensor):
        """
        Regression head follows steps of
            - compute scale for disparity (if there is downsampling)
            - impose uniqueness constraint by optimal transport
            - compute RR loss
            - regress disparity and occlusion
            - upsample (if there is downsampling) and adjust based on context
        
        :param attn_weight: raw attention weight, [N,H,W,W]
        :param x: input data
        :return: dictionary of predicted values
        """
        bs, _, h, w = x.left.size()
        output = {}

        # compute scale
        if x.sampled_cols is not None:
            scale = x.left.size(-1) / float(x.sampled_cols.size(-1))
        else:
            scale = 1.0

        # normalize attention to 0-1
        if self.ot:
            # optimal transport
            attn_ot = self._optimal_transport(attn_weight, 10)
        else:
            # softmax
            attn_ot = self._softmax(attn_weight)

        # compute relative response (RR) at ground truth location
        if x.disp is not None:
            # find ground truth response (gt_response) and location (target)
            output['gt_response'], target = self._compute_gt_location(scale, x.sampled_cols, x.sampled_rows,
                                                                      attn_ot[..., :-1, :-1], x.disp)
        else:
            output['gt_response'] = None

        # compute relative response (RR) at occluded location
        if x.occ_mask is not None:
            # handle occlusion
            occ_mask = x.occ_mask
            occ_mask_right = x.occ_mask_right
            if x.sampled_cols is not None:
                occ_mask = batched_index_select(occ_mask, 2, x.sampled_cols)
                occ_mask_right = batched_index_select(occ_mask_right, 2, x.sampled_cols)
            if x.sampled_rows is not None:
                occ_mask = batched_index_select(occ_mask, 1, x.sampled_rows)
                occ_mask_right = batched_index_select(occ_mask_right, 1, x.sampled_rows)

            output['gt_response_occ_left'] = attn_ot[..., :-1, -1][occ_mask]
            output['gt_response_occ_right'] = attn_ot[..., -1, :-1][occ_mask_right]
        else:
            output['gt_response_occ_left'] = None
            output['gt_response_occ_right'] = None
            occ_mask = x.occ_mask

        # regress low res disparity
        pos_shift = self._compute_unscaled_pos_shift(attn_weight.shape[2], attn_weight.device)  # NxHxW_leftxW_right
        disp_pred_low_res, matched_attn = self._compute_low_res_disp(pos_shift, attn_ot[..., :-1, :-1], occ_mask)
        # regress low res occlusion
        occ_pred_low_res = self._compute_low_res_occ(matched_attn)

        # with open('attn_weight.dat', 'wb') as f:
        #     torch.save(attn_ot[0], f)
        # with open('target.dat', 'wb') as f:
        #     torch.save(target, f)

        # upsample and context adjust
        if x.sampled_cols is not None:
            output['disp_pred'], output['disp_pred_low_res'], output['occ_pred'] = self._upsample(x, disp_pred_low_res,
                                                                                                  occ_pred_low_res,
                                                                                                  scale)
        else:
            output['disp_pred'] = disp_pred_low_res
            output['occ_pred'] = occ_pred_low_res

        return output


def build_regression_head(args):
    cal = build_context_adjustment_layer(args)

    if args.regression_head == 'ot':
        ot = True
    elif args.regression_head == 'softmax':
        ot = False
    else:
        raise Exception('Regression head type not recognized: ', args.regression_head)

    return RegressionHead(cal, ot)

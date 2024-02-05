#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import time

import torch
from tqdm import tqdm

from utilities.misc import NestedTensor, save_and_clear


def forward_pass_without_loss(model, data, device, downsample):
    # read data
    left, right = data['left'].to(device), data['right'].to(device)

    # we uniformly sample for training
    bs, _, h, w = left.size()
    if downsample <= 0:
        sampled_cols = None
        sampled_rows = None
    else:
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).to(device)
        sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).to(device)

    inputs = NestedTensor(left, right, sampled_cols=sampled_cols, sampled_rows=sampled_rows)

    # forward pass
    start = time.time()
    outputs = model(inputs)
    end = time.time()
    time_elapse = end - start

    return outputs, time_elapse


@torch.no_grad()
def inference(model, data_loader, device, downsample):
    output_idx = 0

    model.eval()

    tbar = tqdm(data_loader)

    output_file = {'left': [], 'right': [], 'disp_pred': [], 'occ_pred': [], 'time': []}

    for idx, data in enumerate(tbar):
        # forward pass
        outputs, time_elapse = forward_pass_without_loss(model, data, device, downsample)

        # save output
        output_file['left'].append(data['left'][0])
        output_file['right'].append(data['right'][0])
        output_file['disp_pred'].append(outputs['disp_pred'].data[0].cpu())
        output_file['occ_pred'].append(outputs['occ_pred'].data[0].cpu())
        output_file['time'].append(time_elapse)

        print("disparity", outputs['disp_pred'].max().item(), outputs['disp_pred'].min().item())

        # save to file
        if len(output_file['left']) >= 50:
            output_idx = save_and_clear(output_idx, output_file)

    # save to file
    save_and_clear(output_idx, output_file)

    return

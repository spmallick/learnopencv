#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

from typing import Iterable

import torch
from tqdm import tqdm

from utilities.foward_pass import forward_pass, write_summary
from utilities.misc import save_and_clear
from utilities.summary_logger import TensorboardSummary


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, device: torch.device,
             epoch: int, summary: TensorboardSummary, save_output: bool):
    model.eval()
    criterion.eval()

    # initialize stats
    eval_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                  'total_px': 0.0}
    # config text logger
    logger = summary.config_logger(epoch)
    # init output file
    if save_output:
        output_idx = 0
        output_file = {'left': [], 'right': [], 'disp': [], 'disp_pred': [], 'occ_mask': [], 'occ_pred': []}

    tbar = tqdm(data_loader)
    valid_samples = len(tbar)
    for idx, data in enumerate(tbar):
        # forward pass
        outputs, losses, sampled_disp = forward_pass(model, data, device, criterion, eval_stats, idx, logger)

        if losses is None:
            valid_samples -= 1
            continue

        # clear cache
        torch.cuda.empty_cache()

        # save output
        if save_output:
            output_file['left'].append(data['left'][0])
            output_file['right'].append(data['right'][0])
            output_file['disp'].append(data['disp'][0])
            output_file['occ_mask'].append(data['occ_mask'][0].cpu())
            output_file['disp_pred'].append(outputs['disp_pred'].data[0].cpu())
            output_file['occ_pred'].append(outputs['occ_pred'].data[0].cpu())

            # save to file
            if len(output_file['left']) >= 50:
                output_idx = save_and_clear(output_idx, output_file)

    # save to file
    if save_output:
        save_and_clear(output_idx, output_file)

    # compute avg
    eval_stats['epe'] = eval_stats['epe'] / valid_samples
    eval_stats['iou'] = eval_stats['iou'] / valid_samples
    eval_stats['px_error_rate'] = eval_stats['error_px'] / eval_stats['total_px']

    # write to tensorboard
    write_summary(eval_stats, summary, epoch, 'eval')

    # log to text
    logger.info('Epoch %d, epe %.4f, iou %.4f, px error %.4f' %
                (epoch, eval_stats['epe'], eval_stats['iou'], eval_stats['px_error_rate']))
    print()

    return eval_stats

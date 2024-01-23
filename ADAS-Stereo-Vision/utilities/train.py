#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import math
import sys
from typing import Iterable

import torch
from tqdm import tqdm

from utilities.foward_pass import forward_pass, write_summary
from utilities.summary_logger import TensorboardSummary


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.Module, device: torch.device, epoch: int, summary: TensorboardSummary,
                    max_norm: float = 0, amp: object = None):
    """
    train model for 1 epoch
    """
    model.train()
    criterion.train()

    # initialize stats
    train_stats = {'l1': 0.0, 'occ_be': 0.0, 'l1_raw': 0.0, 'iou': 0.0, 'rr': 0.0, 'epe': 0.0, 'error_px': 0.0,
                   'total_px': 0.0}

    tbar = tqdm(data_loader)
    for idx, data in enumerate(tbar):
        # forward pass
        _, losses, sampled_disp = forward_pass(model, data, device, criterion, train_stats)

        if losses is None:
            continue

        # terminate training if exploded
        if not math.isfinite(losses['aggregated'].item()):
            print("Loss is {}, stopping training".format(losses['aggregated'].item()))
            sys.exit(1)

        # backprop
        optimizer.zero_grad()
        if amp is not None:
            with amp.scale_loss(losses['aggregated'], optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses['aggregated'].backward()

        # clip norm
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # step optimizer
        optimizer.step()

        print('pixel_error', losses['error_px'] / losses['total_px'])

        # clear cache
        torch.cuda.empty_cache()

    # compute avg
    train_stats['px_error_rate'] = train_stats['error_px'] / train_stats['total_px']

    # log to tensorboard
    write_summary(train_stats, summary, epoch, 'train')

    print('Training loss', train_stats['l1'], 'pixel error rate', train_stats['px_error_rate'])
    print('RR loss', train_stats['rr'])

    return

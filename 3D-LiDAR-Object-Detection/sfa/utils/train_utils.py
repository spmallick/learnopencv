"""
'''
///////////////////////////////////////
3D LiDAR Object Detection - ADAS
Pranav Durai
//////////////////////////////////////
'''
# Description: utils functions that use for training process
"""

import copy
import os
import math
import sys

import torch
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.lr_scheduler import OneCyclePolicy


def create_optimizer(configs, model):
    """Create optimizer for training process
    """
    if hasattr(model, 'module'):
        train_params = [param for param in model.module.parameters() if param.requires_grad]
    else:
        train_params = [param for param in model.parameters() if param.requires_grad]

    if configs.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=configs.lr, momentum=configs.momentum, nesterov=True)
    elif configs.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=configs.lr, weight_decay=configs.weight_decay)
    else:
        assert False, "Unknown optimizer type"

    return optimizer


def create_lr_scheduler(optimizer, configs):
    """Create learning rate scheduler for training process"""

    if configs.lr_type == 'multi_step':
        def multi_step_scheduler(i):
            if i < configs.steps[0]:
                factor = 1.
            elif i < configs.steps[1]:
                factor = 0.1
            else:
                factor = 0.01

            return factor

        lr_scheduler = LambdaLR(optimizer, multi_step_scheduler)

    elif configs.lr_type == 'cosin':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: (((1 + math.cos(x * math.pi / configs.num_epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
    elif configs.lr_type == 'one_cycle':
        lr_scheduler = OneCyclePolicy(optimizer, configs.lr, configs.num_epochs, momentum_rng=[0.85, 0.95],
                                      phase_ratio=0.45)
    else:
        raise ValueError

    plot_lr_scheduler(optimizer, lr_scheduler, configs.num_epochs, save_dir=configs.logs_dir, lr_type=configs.lr_type)

    return lr_scheduler


def get_saved_state(model, optimizer, lr_scheduler, epoch, configs):
    """Get the information to save with checkpoints"""
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    utils_state_dict = {
        'epoch': epoch,
        'configs': configs,
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'lr_scheduler': copy.deepcopy(lr_scheduler.state_dict())
    }

    return model_state_dict, utils_state_dict


def save_checkpoint(checkpoints_dir, saved_fn, model_state_dict, utils_state_dict, epoch):
    """Save checkpoint every epoch only is best model or after every checkpoint_freq epoch"""
    model_save_path = os.path.join(checkpoints_dir, 'Model_{}_epoch_{}.pth'.format(saved_fn, epoch))
    utils_save_path = os.path.join(checkpoints_dir, 'Utils_{}_epoch_{}.pth'.format(saved_fn, epoch))

    torch.save(model_state_dict, model_save_path)
    torch.save(utils_state_dict, utils_save_path)

    print('save a checkpoint at {}'.format(model_save_path))


def plot_lr_scheduler(optimizer, scheduler, num_epochs=300, save_dir='', lr_type=''):
    # Plot LR simulating training for full num_epochs
    optimizer, scheduler = copy.copy(optimizer), copy.copy(scheduler)  # do not modify originals
    y = []
    for _ in range(num_epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, num_epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'LR_{}.png'.format(lr_type)), dpi=200)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    from torchvision.models import resnet18

    configs = edict()
    configs.steps = [150, 180]
    configs.lr_type = 'one_cycle'  # multi_step, cosin, one_csycle
    configs.logs_dir = '../../logs/'
    configs.num_epochs = 50
    configs.lr = 2.25e-3
    net = resnet18()
    optimizer = torch.optim.Adam(net.parameters(), 0.0002)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
    scheduler = create_lr_scheduler(optimizer, configs)
    for i in range(configs.num_epochs):
        print(i, scheduler.get_lr())
        scheduler.step()

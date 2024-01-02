"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.08.09
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: utils functions that use for model
"""

import os
import sys

import torch

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from models import resnet, fpn_resnet


def create_model(configs):
    """Create model based on architecture name"""
    try:
        arch_parts = configs.arch.split('_')
        num_layers = int(arch_parts[-1])
    except:
        raise ValueError
    if 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)
    elif 'resnet' in configs.arch:
        print('using ResNet architecture')
        model = resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                    imagenet_pretrained=configs.imagenet_pretrained)
    else:
        assert False, 'Undefined model backbone'

    return model


def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model


if __name__ == '__main__':
    import argparse

    from torchsummary import summary
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(description='RTM3D Implementation')
    parser.add_argument('-a', '--arch', type=str, default='resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--head_conv', type=int, default=-1,
                        help='conv layer channels for output head'
                             '0 for no conv layer'
                             '-1 for default setting: '
                             '64 for resnets and 256 for dla.')

    configs = edict(vars(parser.parse_args()))
    if configs.head_conv == -1:  # init default head_conv
        configs.head_conv = 256 if 'dla' in configs.arch else 64

    configs.num_classes = 3
    configs.num_vertexes = 8
    configs.num_center_offset = 2
    configs.num_vertexes_offset = 2
    configs.num_dimension = 3
    configs.num_rot = 8
    configs.num_depth = 1
    configs.num_wh = 2
    configs.heads = {
        'hm_mc': configs.num_classes,
        'hm_ver': configs.num_vertexes,
        'vercoor': configs.num_vertexes * 2,
        'cenoff': configs.num_center_offset,
        'veroff': configs.num_vertexes_offset,
        'dim': configs.num_dimension,
        'rot': configs.num_rot,
        'depth': configs.num_depth,
        'wh': configs.num_wh
    }

    configs.device = torch.device('cuda:1')
    # configs.device = torch.device('cpu')

    model = create_model(configs).to(device=configs.device)
    sample_input = torch.randn((1, 3, 224, 224)).to(device=configs.device)
    # summary(model.cuda(1), (3, 224, 224))
    output = model(sample_input)
    for hm_name, hm_out in output.items():
        print('hm_name: {}, hm_out size: {}'.format(hm_name, hm_out.size()))

    print('number of parameters: {}'.format(get_num_parameters(model)))

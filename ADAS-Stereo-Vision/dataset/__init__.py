#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch.utils.data as data

from dataset.kitti import KITTI2015Dataset, KITTI2012Dataset, KITTIDataset
from dataset.middlebury import Middlebury2014Dataset
from dataset.scared import ScaredDataset
from dataset.scene_flow import SceneFlowSamplePackDataset, SceneFlowFlyingThingsDataset, SceneFlowMonkaaDataset
from dataset.sintel import SintelDataset


def build_data_loader(args):
    '''
    Build data loader

    :param args: arg parser object
    :return: train, validation and test dataloaders
    '''
    if args.dataset_directory == '':
        raise ValueError(f'Dataset directory cannot be empty.')
    else:
        dataset_dir = args.dataset_directory

    if args.dataset == 'sceneflow':
        dataset_train = SceneFlowFlyingThingsDataset(dataset_dir, 'train')
        dataset_validation = SceneFlowFlyingThingsDataset(dataset_dir, args.validation)
        dataset_test = SceneFlowFlyingThingsDataset(dataset_dir, 'test')
    elif args.dataset == 'sceneflow_monkaa':
        dataset_train = SceneFlowMonkaaDataset(dataset_dir, 'train')
        dataset_validation = SceneFlowMonkaaDataset(dataset_dir, args.validation)
        dataset_test = SceneFlowMonkaaDataset(dataset_dir, 'test')
    elif args.dataset == 'kitti2015':
        dataset_train = KITTI2015Dataset(dataset_dir, 'train')
        dataset_validation = KITTI2015Dataset(dataset_dir, args.validation)
        dataset_test = KITTI2015Dataset(dataset_dir, 'test')
    elif args.dataset == 'kitti2012':
        dataset_train = KITTI2012Dataset(dataset_dir, 'train')
        dataset_validation = KITTI2012Dataset(dataset_dir, args.validation)
        dataset_test = KITTI2012Dataset(dataset_dir, 'test')
    elif args.dataset == 'kitti':
        dataset_train = KITTIDataset(dataset_dir, split='train')
        dataset_validation = KITTIDataset(dataset_dir, split=args.validation)
        dataset_test = KITTIDataset(dataset_dir, split='test')
    elif args.dataset == 'middlebury2014':
        dataset_train = Middlebury2014Dataset(dataset_dir, 'train')
        dataset_validation = Middlebury2014Dataset(dataset_dir, args.validation)
        dataset_test = Middlebury2014Dataset(dataset_dir, 'test')
    elif args.dataset == 'scared':
        dataset_train = ScaredDataset(dataset_dir, 'train')
        dataset_validation = ScaredDataset(dataset_dir, args.validation)
        dataset_test = ScaredDataset(dataset_dir, 'test')
    elif args.dataset == 'sintel':
        dataset_train = SintelDataset(dataset_dir, 'train')
        dataset_validation = SintelDataset(dataset_dir, args.validation)
        dataset_test = SintelDataset(dataset_dir, 'test')

    elif args.dataset == 'sceneflow_toy':
        dataset_train = SceneFlowSamplePackDataset(dataset_dir, 'train')
        dataset_validation = SceneFlowSamplePackDataset(dataset_dir, 'validation')
        dataset_test = SceneFlowSamplePackDataset(dataset_dir, 'validation')
    elif args.dataset == 'kitti_toy':
        dataset_train = KITTI2015Dataset(dataset_dir, 'train')
        dataset_validation = KITTI2015Dataset(dataset_dir, 'validation')
        dataset_test = KITTI2015Dataset(dataset_dir, 'validation')
    elif args.dataset == 'middlebury_toy':
        dataset_train = Middlebury2014Dataset(dataset_dir, 'train')
        dataset_validation = Middlebury2014Dataset(dataset_dir, 'validation')
        dataset_test = Middlebury2014Dataset(dataset_dir, 'validation')
    elif args.dataset == 'scared_toy':
        dataset_train = ScaredDataset(dataset_dir, 'train')
        dataset_validation = ScaredDataset(dataset_dir, 'validation')
        dataset_test = ScaredDataset(dataset_dir, 'validation')

    else:
        raise ValueError(f'Dataset not recognized: {args.dataset}')

    data_loader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)
    data_loader_validation = data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    data_loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, pin_memory=True)

    return data_loader_train, data_loader_validation, data_loader_test

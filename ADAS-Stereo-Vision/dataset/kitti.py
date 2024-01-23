#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations import Compose
from natsort import natsorted

from dataset.preprocess import augment, normalization
from dataset.stereo_albumentation import RGBShiftStereo, RandomBrightnessContrastStereo, random_crop


class KITTIBaseDataset(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(KITTIBaseDataset, self).__init__()

        self.datadir = datadir
        self.split = split

        if split == 'train' or split == 'validation' or split == 'validation_all':
            self.sub_folder = 'training/'
        elif split == 'test':
            self.sub_folder = 'testing/'

        # to be set by child classes
        self.left_fold = None
        self.right_fold = None
        self.disp_fold = None

        self._augmentation()

    def _read_data(self):
        assert self.left_fold is not None

        self.left_data = natsorted([os.path.join(self.datadir, self.sub_folder, self.left_fold, img) for img in
                                    os.listdir(os.path.join(self.datadir, self.sub_folder, self.left_fold)) if
                                    img.find('_10') > -1])
        self.right_data = [img.replace(self.left_fold, self.right_fold) for img in self.left_data]
        self.disp_data = [img.replace(self.left_fold, self.disp_fold) for img in self.left_data]

        self._split_data()

    def _split_data(self):
        train_val_frac = 0.95
        # split data
        if len(self.left_data) > 1:
            if self.split == 'train':
                self.left_data = self.left_data[:int(len(self.left_data) * train_val_frac)]
                self.right_data = self.right_data[:int(len(self.right_data) * train_val_frac)]
                self.disp_data = self.disp_data[:int(len(self.disp_data) * train_val_frac)]
            elif self.split == 'validation':
                self.left_data = self.left_data[int(len(self.left_data) * train_val_frac):]
                self.right_data = self.right_data[int(len(self.right_data) * train_val_frac):]
                self.disp_data = self.disp_data[int(len(self.disp_data) * train_val_frac):]

    def _augmentation(self):
        if self.split == 'train':
            self.transformation = Compose([
                RGBShiftStereo(always_apply=True, p_asym=0.5),
                RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
            ])
        elif self.split == 'validation' or self.split == 'test' or self.split == 'validation_all':
            self.transformation = None
        else:
            raise Exception("Split not recognized")

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        input_data = {}

        # left
        left_fname = self.left_data[idx]
        left = np.array(Image.open(left_fname)).astype(np.uint8)
        input_data['left'] = left

        # right
        right_fname = self.right_data[idx]
        right = np.array(Image.open(right_fname)).astype(np.uint8)
        input_data['right'] = right

        # disp
        if not self.split == 'test':  # no disp for test files
            disp_fname = self.disp_data[idx]

            disp = np.array(Image.open(disp_fname)).astype(float) / 256.
            input_data['disp'] = disp
            input_data['occ_mask'] = np.zeros_like(disp).astype(bool)

            if self.split == 'train':
                input_data = random_crop(200, 640, input_data, self.split)

            input_data = augment(input_data, self.transformation)
        else:
            input_data = normalization(**input_data)

        return input_data


class KITTI2015Dataset(KITTIBaseDataset):
    def __init__(self, datadir, split='train'):
        super(KITTI2015Dataset, self).__init__(datadir, split)

        self.left_fold = 'image_2/'
        self.right_fold = 'image_3/'
        self.disp_fold = 'disp_occ_0/'  # we read disp data with occlusion since we compute occ directly

        self._read_data()


class KITTI2012Dataset(KITTIBaseDataset):
    def __init__(self, datadir, split='train'):
        super(KITTI2012Dataset, self).__init__(datadir, split)

        self.left_fold = 'colored_0/'
        self.right_fold = 'colored_1/'
        self.disp_fold = 'disp_occ/'  # we read disp data with occlusion since we compute occ directly

        self._read_data()


class KITTIDataset(KITTIBaseDataset):
    """
    Merged KITTI dataset with 2015 and 2012 data
    """

    def __init__(self, datadir, split='train'):
        super(KITTIDataset, self).__init__(datadir, split)

        self.left_fold_2015 = 'image_2'
        self.right_fold_2015 = 'image_3'
        self.disp_fold_2015 = 'disp_occ_0'  # we read disp data with occlusion since we compute occ directly
        self.preprend_2015 = '2015'

        self.left_fold_2012 = 'colored_0'
        self.right_fold_2012 = 'colored_1'
        self.disp_fold_2012 = 'disp_occ'  # we we read disp data with occlusion since we compute occ directly
        self.preprend_2012 = '2012'

        self._read_data()

    def _read_data(self):
        assert self.left_fold_2015 is not None
        assert self.left_fold_2012 is not None

        left_data_2015 = [os.path.join(self.datadir, self.preprend_2015, self.sub_folder, self.left_fold_2015, img) for
                          img in os.listdir(os.path.join(self.datadir, '2015', self.sub_folder, self.left_fold_2015)) if
                          img.find('_10') > -1]
        left_data_2015 = natsorted(left_data_2015)
        right_data_2015 = [img.replace(self.left_fold_2015, self.right_fold_2015) for img in left_data_2015]
        disp_data_2015 = [img.replace(self.left_fold_2015, self.disp_fold_2015) for img in left_data_2015]

        left_data_2012 = [os.path.join(self.datadir, self.preprend_2012, self.sub_folder, self.left_fold_2012, img) for
                          img in os.listdir(os.path.join(self.datadir, '2012', self.sub_folder, self.left_fold_2012)) if
                          img.find('_10') > -1]
        left_data_2012 = natsorted(left_data_2012)
        right_data_2012 = [img.replace(self.left_fold_2012, self.right_fold_2012) for img in left_data_2012]
        disp_data_2012 = [img.replace(self.left_fold_2012, self.disp_fold_2012) for img in left_data_2012]

        self.left_data = natsorted(left_data_2015 + left_data_2012)
        self.right_data = natsorted(right_data_2015 + right_data_2012)
        self.disp_data = natsorted(disp_data_2015 + disp_data_2012)

        self._split_data()

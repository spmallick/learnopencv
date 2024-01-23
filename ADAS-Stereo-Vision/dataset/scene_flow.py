#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import os

import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations import Compose, OneOf
from natsort import natsorted

from dataset.preprocess import augment
from dataset.stereo_albumentation import RandomShiftRotate, GaussNoiseStereo, RGBShiftStereo, \
    RandomBrightnessContrastStereo, random_crop, horizontal_flip
from utilities.python_pfm import readPFM


class SceneFlowSamplePackDataset(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(SceneFlowSamplePackDataset, self).__init__()

        self.datadir = datadir
        self.left_fold = 'RGB_cleanpass/left/'
        self.right_fold = 'RGB_cleanpass/right/'
        self.disp = 'disparity/left'
        self.disp_right = 'disparity/right'
        self.occ_fold = 'occlusion/left'
        self.occ_fold_right = 'occlusion/right'

        self.data = os.listdir(os.path.join(self.datadir, self.left_fold))

        self._augmentation()

    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = {}

        path = self.datadir

        left = np.array(Image.open(os.path.join(path, self.left_fold, self.data[idx]))).astype(np.uint8)[..., :3]
        input_data['left'] = left

        right = np.array(Image.open(os.path.join(path, self.right_fold, self.data[idx]))).astype(np.uint8)[..., :3]
        input_data['right'] = right

        occ = np.array(Image.open(os.path.join(path, self.occ_fold, self.data[idx]))).astype(np.bool)
        input_data['occ_mask'] = occ

        occ_right = np.array(Image.open(os.path.join(path, self.occ_fold_right, self.data[idx]))).astype(np.bool)
        input_data['occ_mask_right'] = occ_right

        disp, _ = readPFM(os.path.join(path, self.disp, self.data[idx].replace('png', 'pfm')))
        input_data['disp'] = disp

        disp_right, _ = readPFM(os.path.join(path, self.disp_right, self.data[idx].replace('png', 'pfm')))
        input_data['disp_right'] = disp_right

        input_data = augment(input_data, self.transformation)

        return input_data


class SceneFlowFlyingThingsDataset(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(SceneFlowFlyingThingsDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        if self.split == 'train':
            self.split_folder = 'TRAIN'
        else:
            self.split_folder = 'TEST'

        self._read_data()
        self._augmentation()

    def _read_data(self):
        directory = os.path.join(self.datadir, 'frame_finalpass', self.split_folder)
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

        seq_folders = []
        for sub_folder in sub_folders:
            seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                            os.path.isdir(os.path.join(sub_folder, seq))]

        self.left_data = []
        for seq_folder in seq_folders:
            self.left_data += [os.path.join(seq_folder, 'left', img) for img in
                               os.listdir(os.path.join(seq_folder, 'left'))]

        self.left_data = natsorted(self.left_data)

        directory = os.path.join(self.datadir, 'occlusion', self.split_folder, 'left')
        self.occ_data = [os.path.join(directory, occ) for occ in os.listdir(directory)]
        self.occ_data = natsorted(self.occ_data)

    def _augmentation(self):
        if self.split == 'train':
            self.transformation = Compose([
                RandomShiftRotate(always_apply=True),
                RGBShiftStereo(always_apply=True, p_asym=0.3),
                OneOf([
                    GaussNoiseStereo(always_apply=True, p_asym=1.0),
                    RandomBrightnessContrastStereo(always_apply=True, p_asym=0.5)
                ], p=1.0)
            ])
        else:
            self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        result = {}

        left_fname = self.left_data[idx]
        result['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]

        right_fname = left_fname.replace('left', 'right')
        result['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]

        occ_right_fname = self.occ_data[idx].replace('left', 'right')
        occ_left = np.array(Image.open(self.occ_data[idx])).astype(np.bool)
        occ_right = np.array(Image.open(occ_right_fname)).astype(np.bool)

        disp_left_fname = left_fname.replace('frame_finalpass', 'disparity').replace('.png', '.pfm')
        disp_right_fname = right_fname.replace('frame_finalpass', 'disparity').replace('.png', '.pfm')
        disp_left, _ = readPFM(disp_left_fname)
        disp_right, _ = readPFM(disp_right_fname)

        if self.split == "train":
            # horizontal flip
            result['left'], result['right'], result['occ_mask'], result['occ_mask_right'], disp, disp_right \
                = horizontal_flip(result['left'], result['right'], occ_left, occ_right, disp_left, disp_right,
                                  self.split)
            result['disp'] = np.nan_to_num(disp, nan=0.0)
            result['disp_right'] = np.nan_to_num(disp_right, nan=0.0)

            # random crop        
            result = random_crop(360, 640, result, self.split)
        else:
            result['occ_mask'] = occ_left
            result['occ_mask_right'] = occ_right
            result['disp'] = disp_left
            result['disp_right'] = disp_right

        result = augment(result, self.transformation)

        return result


class SceneFlowMonkaaDataset(data.Dataset):
    def __init__(self, datadir, split='train'):
        super(SceneFlowMonkaaDataset, self).__init__()

        self.datadir = datadir
        self.split = split
        self._read_data()
        self._augmentation()

    def _read_data(self):
        directory = os.path.join(self.datadir, 'frames_cleanpass')
        sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

        self.left_data = []
        for sub_folder in sub_folders:
            self.left_data += [os.path.join(sub_folder, 'left', img) for img in
                               os.listdir(os.path.join(sub_folder, 'left'))]

        self.left_data = natsorted(self.left_data)

    def _split_data(self):
        return

    def _augmentation(self):
        self.transformation = None

    def __len__(self):
        return len(self.left_data)

    def __getitem__(self, idx):
        result = {}

        left_fname = self.left_data[idx]
        result['left'] = np.array(Image.open(left_fname)).astype(np.uint8)[..., :3]

        right_fname = left_fname.replace('left', 'right')
        result['right'] = np.array(Image.open(right_fname)).astype(np.uint8)[..., :3]

        disp_left_fname = left_fname.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
        disp_right_fname = right_fname.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
        disp_left, _ = readPFM(disp_left_fname)
        disp_right, _ = readPFM(disp_right_fname)

        occ_left_fname = left_fname.replace('frames_cleanpass', 'occlusion')
        occ_right_fname = right_fname.replace('frames_cleanpass', 'occlusion')
        occ_left = np.array(Image.open(occ_left_fname)).astype(np.bool)
        occ_right = np.array(Image.open(occ_right_fname)).astype(np.bool)

        result['occ_mask'] = occ_left
        result['occ_mask_right'] = occ_right
        result['disp'] = disp_left
        result['disp_right'] = disp_right

        result = augment(result, self.transformation)

        return result

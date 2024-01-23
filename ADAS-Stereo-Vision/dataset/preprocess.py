#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import numpy as np
import torch
from albumentations import Compose

from dataset.stereo_albumentation import Normalize, ToTensor

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

normalization = Compose([Normalize(always_apply=True),
                         ToTensor(always_apply=True)], p=1.0)


def denormalize(img):
    """
    De-normalize a tensor and return img

    :param img: normalized image, [C,H,W]
    :return: original image, [H,W,C]
    """

    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0)  # H,W,C
        img *= torch.tensor(__imagenet_stats['std'])
        img += torch.tensor(__imagenet_stats['mean'])
        return img.numpy()
    else:
        img = img.transpose(1, 2, 0)  # H,W,C
        img *= np.array(__imagenet_stats['std'])
        img += np.array(__imagenet_stats['mean'])
        return img


def compute_left_occ_region(w, disp):
    """
    Compute occluded region on the left image border

    :param w: image width
    :param disp: left disparity
    :return: occ mask
    """

    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    shifted_coord = coord - disp
    occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ

    return occ_mask


def compute_right_occ_region(w, disp):
    """
    Compute occluded region on the right image border

    :param w: image width
    :param disp: right disparity
    :return: occ mask
    """
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    shifted_coord = coord + disp
    occ_mask = shifted_coord > w  # occlusion mask, 1 indicates occ

    return occ_mask


def augment(input_data, transformation):
    """
    apply augmentation and find occluded pixels
    """

    if transformation is not None:
        # perform augmentation first
        input_data = transformation(**input_data)

    w = input_data['disp'].shape[-1]
    # set large/small values to be 0
    input_data['disp'][input_data['disp'] > w] = 0
    input_data['disp'][input_data['disp'] < 0] = 0

    # manually compute occ area (this is necessary after cropping)
    occ_mask = compute_left_occ_region(w, input_data['disp'])
    input_data['occ_mask'][occ_mask] = True  # update
    input_data['occ_mask'] = np.ascontiguousarray(input_data['occ_mask'])

    # manually compute occ area for right image
    try:
        occ_mask = compute_right_occ_region(w, input_data['disp_right'])
        input_data['occ_mask_right'][occ_mask] = 1
        input_data['occ_mask_right'] = np.ascontiguousarray(input_data['occ_mask_right'])
    except KeyError:
        # print('No disp mask right, check if dataset is KITTI')
        input_data['occ_mask_right'] = np.zeros_like(occ_mask).astype(bool)
    input_data.pop('disp_right', None)  # remove disp right after finish

    # set occlusion area to 0
    occ_mask = input_data['occ_mask']
    input_data['disp'][occ_mask] = 0
    input_data['disp'] = np.ascontiguousarray(input_data['disp'], dtype=np.float32)

    # return normalized image
    return normalization(**input_data)

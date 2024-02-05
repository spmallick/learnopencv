#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import random

import albumentations.augmentations.functional as F
import cv2
import numpy as np
import torch
from albumentations import GaussNoise, RGBShift, RandomBrightnessContrast, ToGray
from albumentations.core.transforms_interface import BasicTransform

"""
functions that cannot fit in albumentation framework
"""


def get_random_crop_coords(height, width, crop_height, crop_width):
    """
    get coordinates for cropping

    :param height: image height, int
    :param width: image width, int
    :param crop_height: crop height, int
    :param crop_width: crop width, int
    :return: xy coordinates
    """
    y1 = random.randint(0, height - crop_height)
    y2 = y1 + crop_height
    x1 = random.randint(0, width - crop_width)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def crop(img, x1, y1, x2, y2):
    """
    crop image given coordinates

    :param img: input image, [H,W,3]
    :param x1: coordinate, int
    :param y1: coordinate, int
    :param x2: coordinate, int
    :param y2: coordinate, int
    :return: cropped image
    """
    img = img[y1:y2, x1:x2]
    return img


def horizontal_flip(img_left, img_right, occ_left, occ_right, disp_left, disp_right, split):
    """
    horizontal flip left and right images, then disparity has to be swapped

    :param img_left: left image, [H,W,3]
    :param img_right: right image, [H,W,3]
    :param occ_left: left occlusion mask, [H,W]
    :param occ_right: right occlusion mask, [H,W]
    :param disp_left: left disparity, [H,W]
    :param disp_right: right disparity, [H,W]
    :param split: train/validation split, string
    :return: updated data
    """
    if split == 'validation':
        p = 0.0
    else:
        p = random.random()

    # if hflip, we flip left/right, and read everything of right images
    if p >= 0.5:
        left_flipped = img_left[:, ::-1]
        right_flipped = img_right[:, ::-1]
        img_left = right_flipped
        img_right = left_flipped

        occ = occ_right[:, ::-1]
        occ_right = occ_left[:, ::-1]
        disp = disp_right[:, ::-1]
        disp_right = disp_left[:, ::-1]
    else:
        occ = occ_left
        disp = disp_left

    return img_left, img_right, occ, occ_right, disp, disp_right


def random_crop(min_crop_height, min_crop_width, input_data, split):
    """
    Crop center part of the input with a random width and height.

    :param min_crop_height: min height of the crop, int
    :param min_crop_width: min width of the crop, int
    :param input_data: input data, dictionary
    :param split: train/validation split, string
    :return: updated input data, dictionary
    """

    if split != 'train':
        return input_data

    height, width = input_data['left'].shape[:2]

    if min_crop_height >= height or min_crop_width > width:
        x1 = 0
        x2 = width - 1
        y1 = 0
        y2 = height - 1
    else:
        crop_height = random.randint(min_crop_height, height)
        crop_width = random.randint(min_crop_width, width)

        x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width)

    input_data['left'] = crop(input_data['left'], x1, y1, x2, y2)
    input_data['right'] = crop(input_data['right'], x1, y1, x2, y2)
    input_data['disp'] = crop(input_data['disp'], x1, y1, x2, y2)
    input_data['occ_mask'] = crop(input_data['occ_mask'], x1, y1, x2, y2)
    try:
        input_data['disp_right'] = crop(input_data['disp_right'], x1, y1, x2, y2)
        input_data['occ_mask_right'] = crop(input_data['occ_mask_right'], x1, y1, x2, y2)
    except KeyError:
        pass

    return input_data


"""
Base
"""


class StereoTransform(BasicTransform):
    """
    Transform applied to image only.
    """

    @property
    def targets(self):
        return {
            "left": self.apply,
            "right": self.apply
        }

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["left"].shape[1], "rows": kwargs["right"].shape[0]})
        return params


class RightOnlyTransform(BasicTransform):
    """
    Transform applied to right image only.
    """

    @property
    def targets(self):
        return {
            "right": self.apply
        }

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["right"].shape[1], "rows": kwargs["right"].shape[0]})
        return params


class StereoTransformAsym(BasicTransform):
    """
    Transform applied not equally to left and right images.
    """

    def __init__(self, always_apply=False, p=0.5, p_asym=0.2):
        super(StereoTransformAsym, self).__init__(always_apply, p)
        self.p_asym = p_asym

    @property
    def targets(self):
        return {
            "left": self.apply_l,
            "right": self.apply_r
        }

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["left"].shape[1], "rows": kwargs["right"].shape[0]})
        return params

    @property
    def targets_as_params(self):
        return ["left", "right"]

    def asym(self):
        return random.random() < self.p_asym
        # return False


"""
Stereo Image only transform
"""


class Normalize(StereoTransform):
    """Divide pixel values by 255 = 2**8 - 1, subtract mean per channel and divide by std per channel.

    Args:
        mean (float, list of float): mean values
        std  (float, list of float): std values
        max_pixel_value (float): maximum possible pixel value

    Targets:
        left, right

    Image types:
        uint8, float32
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                 p=1.0):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def apply(self, image, **params):
        return F.normalize(image, self.mean, self.std, self.max_pixel_value)

    def get_transform_init_args_names(self):
        return ("mean", "std", "max_pixel_value")


class ToTensor(StereoTransform):
    """Change input from HxWxC to CxHxW

    Targets:
        left, right

    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0):
        super(ToTensor, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return torch.tensor(image.transpose(2, 0, 1))


class ToGrayStereo(StereoTransform, ToGray):
    def __init__(self, always_apply=False, p=0.5):
        StereoTransform.__init__(self, always_apply, p)
        ToGray.__init__(self, always_apply, p)


"""
Stereo Image Only Asym Transform
"""


class GaussNoiseStereo(StereoTransformAsym, GaussNoise):
    """Apply gaussian noise to the input image.

    Args:
        var_limit ((float, float) or float): variance range for noise. If var_limit is a single float, the range
            will be (0, var_limit). Default: (10.0, 50.0).
        mean (float): mean of the noise. Default: 0
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5, p_asym=0.2):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        GaussNoise.__init__(self, var_limit, mean, always_apply, p)

    def apply_l(self, img, gauss_l=None, **params):
        return F.gauss_noise(img, gauss=gauss_l)

    def apply_r(self, img, gauss_r=None, **params):
        return F.gauss_noise(img, gauss=gauss_r)

    def get_params_dependent_on_targets(self, params):

        image = params["left"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        gauss_l = random_state.normal(self.mean, sigma, image.shape)

        if self.asym():
            image = params["right"]
            var = random.uniform(self.var_limit[0], self.var_limit[1])
            sigma = var ** 0.5
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

            gauss_r = random_state.normal(self.mean, sigma, image.shape)
        else:
            gauss_r = gauss_l
        return {"gauss_l": gauss_l, "gauss_r": gauss_r}


class RGBShiftStereo(StereoTransformAsym, RGBShift):
    """Randomly shift values for each channel of the input RGB image.

    Args:
        r_shift_limit ((int, int) or int): range for changing values for the red channel. If r_shift_limit is a single
            int, the range will be (-r_shift_limit, r_shift_limit). Default: (-20, 20).
        g_shift_limit ((int, int) or int): range for changing values for the green channel. If g_shift_limit is a
            single int, the range  will be (-g_shift_limit, g_shift_limit). Default: (-20, 20).
        b_shift_limit ((int, int) or int): range for changing values for the blue channel. If b_shift_limit is a single
            int, the range will be (-b_shift_limit, b_shift_limit). Default: (-20, 20).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5, p_asym=0.2):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        RGBShift.__init__(self, r_shift_limit, g_shift_limit, b_shift_limit, always_apply, p)

    def apply_l(self, image, r_shift_l=0, g_shift_l=0, b_shift_l=0, **params):
        return F.shift_rgb(image, r_shift_l, g_shift_l, b_shift_l)

    def apply_r(self, image, r_shift_r=0, g_shift_r=0, b_shift_r=0, **params):
        return F.shift_rgb(image, r_shift_r, g_shift_r, b_shift_r)

    def get_params_dependent_on_targets(self, params):
        r_shift_l = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
        g_shift_l = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
        b_shift_l = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])

        if self.asym():
            r_shift_r = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
            g_shift_r = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
            b_shift_r = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])
        else:
            r_shift_r = r_shift_l
            g_shift_r = g_shift_l
            b_shift_r = b_shift_l

        return {
            "r_shift_l": r_shift_l,
            "g_shift_l": g_shift_l,
            "b_shift_l": b_shift_l,
            "r_shift_r": r_shift_r,
            "g_shift_r": g_shift_r,
            "b_shift_r": b_shift_r,
        }


class RandomBrightnessContrastStereo(StereoTransformAsym, RandomBrightnessContrast):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5,
                 p_asym=0.2):
        StereoTransformAsym.__init__(self, always_apply, p, p_asym)
        RandomBrightnessContrast.__init__(self, brightness_limit, contrast_limit, brightness_by_max, always_apply, p)

    def apply_l(self, img, alpha_l=1.0, beta_l=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha_l, beta_l, self.brightness_by_max)

    def apply_r(self, img, alpha_r=1.0, beta_r=0.0, **params):
        return F.brightness_contrast_adjust(img, alpha_r, beta_r, self.brightness_by_max)

    def get_params_dependent_on_targets(self, params):
        alpha_l = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        beta_l = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])

        if self.asym():
            alpha_r = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
            beta_r = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
        else:
            alpha_r = alpha_l
            beta_r = beta_l

        return {
            "alpha_l": alpha_l,
            "beta_l": beta_l,
            "alpha_r": alpha_r,
            "beta_r": beta_r,
        }


"""
Right Image Only
"""


class RandomShiftRotate(RightOnlyTransform):
    """Randomly apply vertical translate and rotate the input.
    Args:
        max_shift (float): maximum shift in pixels along vertical direction. Default: 1.5.
        max_rotation (float): maximum rotation in degree. Default: 0.2.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    def __init__(self, max_shift=1.5, max_rotation=0.2, always_apply=False, p=1.0):
        super(RandomShiftRotate, self).__init__(always_apply, p)
        self.max_shift = max_shift
        self.max_rotation = max_rotation

    def apply(self, img, **params):
        h, w, _ = img.shape
        shift = random.random() * self.max_shift * 2 - self.max_shift
        rotation = random.random() * self.max_rotation * 2 - self.max_rotation

        matrix = np.float32([[np.cos(np.deg2rad(rotation)), -np.sin(np.deg2rad(rotation)), 0],
                             [np.sin(np.deg2rad(rotation)), np.cos(np.deg2rad(rotation)), shift]])

        return cv2.warpAffine(img, matrix, (w, h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

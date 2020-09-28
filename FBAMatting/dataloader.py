import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class PredDataset(Dataset):
    """ Reads image and trimap pairs from folder.

    """

    def __init__(self, img_dir, trimap_dir):
        self.img_dir, self.trimap_dir = img_dir, trimap_dir
        self.img_names = [
            x
            for x in os.listdir(self.img_dir)
            if os.path.isfile(os.path.join(self.img_dir, x))
        ]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        trimap_name = img_name[:-3] + "png"

        image = read_image(os.path.join(self.img_dir, img_name))
        trimap = read_trimap(os.path.join(self.trimap_dir, trimap_name))
        pred_dict = {"image": image, "trimap": trimap, "name": img_name}

        return pred_dict


def read_image(name):
    return (cv2.imread(name) / 255.0)[:, :, ::-1]


def read_trimap(name):
    trimap_im = cv2.imread(name, 0) / 255.0
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap

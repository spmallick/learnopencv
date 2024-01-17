"""
'''
///////////////////////////////////////
3D LiDAR Object Detection - ADAS
Pranav Durai
//////////////////////////////////////
'''
# Description: This script for the KITTI dataset
"""

import sys
import os
from builtins import int
from glob import glob

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.kitti_data_utils import get_filtered_lidar
from data_process.kitti_bev_utils import makeBEVMap
import config.kitti_config as cnf


class Demo_KittiDataset(Dataset):
    def __init__(self, configs):
        self.dataset_dir = os.path.join(configs.dataset_dir, configs.foldername, configs.foldername[:10],
                                        configs.foldername)
        self.input_size = configs.input_size
        self.hm_size = configs.hm_size

        self.num_classes = configs.num_classes
        self.max_objects = configs.max_objects

        self.image_dir = os.path.join(self.dataset_dir, "image_02", "data")
        self.lidar_dir = os.path.join(self.dataset_dir, "velodyne_points", "data")
        self.label_dir = os.path.join(self.dataset_dir, "label_2", "data")
        self.sample_id_list = sorted(glob(os.path.join(self.lidar_dir, '*.bin')))
        self.sample_id_list = [float(os.path.basename(fn)[:-4]) for fn in self.sample_id_list]
        self.num_samples = len(self.sample_id_list)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        pass

    def load_bevmap_front(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)
        front_lidar = get_filtered_lidar(lidarData, cnf.boundary)
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        front_bevmap = torch.from_numpy(front_bevmap)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, front_bevmap, img_rgb

    def load_bevmap_front_vs_back(self, index):
        """Load only image for the testing phase"""
        sample_id = int(self.sample_id_list[index])
        img_path, img_rgb = self.get_image(sample_id)
        lidarData = self.get_lidar(sample_id)

        front_lidar = get_filtered_lidar(lidarData, cnf.boundary)
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        front_bevmap = torch.from_numpy(front_bevmap)

        back_lidar = get_filtered_lidar(lidarData, cnf.boundary_back)
        back_bevmap = makeBEVMap(back_lidar, cnf.boundary_back)
        back_bevmap = torch.from_numpy(back_bevmap)

        metadatas = {
            'img_path': img_path,
        }

        return metadatas, front_bevmap, back_bevmap, img_rgb

    def get_image(self, idx):
        img_path = os.path.join(self.image_dir, '{:010d}.png'.format(idx))
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img_path, img

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:010d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
